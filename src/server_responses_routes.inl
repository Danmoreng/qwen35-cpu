// Registered on the same authenticated HTTP server and inference scheduler as chat.
auto stream_response = [](httplib::Response &http_response,
                          std::shared_ptr<server::ResponseRecord> record, std::size_t cursor,
                          const std::shared_ptr<Job> &owner) {
  http_response.set_header("Cache-Control", "no-cache");
  http_response.set_header("X-Accel-Buffering", "no");
  http_response.set_chunked_content_provider(
      "text/event-stream",
      [record, cursor](std::size_t, httplib::DataSink &sink) mutable {
        std::string wire;
        while (record->next(cursor, wire)) {
          if (wire.empty())
            wire = ": heartbeat\n\n";
          if (!sink.write(wire.data(), wire.size()))
            return false;
        }
        sink.done();
        return true;
      },
      [owner](bool ok) {
        if (!ok && owner && !owner->response_record->background)
          owner->cancelled = true;
      });
};

auto create_response = [&](const httplib::Request &req, httplib::Response &res, bool count) {
  if (!authorized(req, res))
    return;
  auto job = std::make_shared<Job>();
  bool stored = false;
  std::string id;
  try {
    auto j = Json::parse(req.body);
    Json history = Json::array();
    if (j.contains("previous_response_id") && !j["previous_response_id"].is_null()) {
      auto parent = responses.find(j.at("previous_response_id").get<std::string>());
      if (!parent) {
        send(res, failure(404, "Previous response not found"));
        return;
      }
      const auto previous = parent->snapshot();
      server::require(previous["status"] == "completed" || previous["status"] == "incomplete",
                      "Previous response has no finalized output");
      history = parent->history;
      for (const auto &item : previous["output"])
        history.push_back(item);
      if (!j.contains("input"))
        j["input"] = Json::array();
    }
    if (j.contains("input") && j["input"].is_array()) {
      server::require(j["input"].size() + history.size() <= 1024, "Too many input items");
      std::size_t referenced_bytes = history.dump().size();
      for (auto &item : j["input"]) {
        if (!item.is_object() || item.value("type", std::string()) != "item_reference")
          continue;
        server::fields(item, {"type", "id"});
        auto stored_item = responses.find_item(item.at("id").get<std::string>());
        if (!stored_item) {
          send(res, failure(404, "Referenced item not found"));
          return;
        }
        referenced_bytes += stored_item->dump().size();
        server::require(referenced_bytes <= 262144, "Referenced input exceeds the request size limit");
        item = std::move(*stored_item);
      }
    }
    if (count && !j.contains("model"))
      j["model"] = model_name;
    auto prepared =
        server::prepare_response(std::move(j), std::move(history), config.max_context, model_name);
    static_cast<server::Request &>(*job) = std::move(prepared.request);
    job->truncation_messages = std::move(prepared.formatted_messages);
    if (count) {
      job->count_only = true;
      auto result = job->promise.get_future();
      if (!service.enqueue(job)) {
        send(res, failure(429, "Request capacity exhausted"));
        return;
      }
      auto reply = result.get();
      if (reply.status == 200)
        reply.body["object"] = "response.input_tokens";
      send(res, reply);
      return;
    }
    id = prepared.body.at("id");
    const auto record = std::make_shared<server::ResponseRecord>(prepared);
    job->response_record = record;
    if (prepared.store) {
      if (!responses.insert(id, record)) {
        send(res, failure(429, "Stored response capacity exhausted"));
        return;
      }
      stored = true;
    }
    auto ready = job->admitted.get_future();
    auto result = job->promise.get_future();
    if (!service.enqueue(job)) {
      if (stored)
        responses.erase(id);
      send(res, failure(429, "Request capacity exhausted"));
      return;
    }
    auto await = [&](auto &future) {
      while (future.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready)
        if (req.is_connection_closed()) {
          if (!prepared.background)
            job->cancelled = true;
          return false;
        }
      return true;
    };
    if (!await(ready))
      return;
    const auto admission = ready.get();
    if (admission.status != 200) {
      if (stored)
        responses.erase(id);
      send(res, admission);
      return;
    }
    if (prepared.stream) {
      stream_response(res, record, 0, job);
      return;
    }
    if (!prepared.background) {
      if (!await(result))
        return;
      const auto reply = result.get();
      if (reply.status != 200) {
        send(res, reply);
        return;
      }
    }
    send(res, {200, record->snapshot()});
  } catch (const std::exception &e) {
    send(res, failure(400, e.what()));
  }
};
http.Post("/v1/responses", [&](const auto &req, auto &res) { create_response(req, res, false); });
http.Post("/v1/responses/input_tokens",
          [&](const auto &req, auto &res) { create_response(req, res, true); });
http.Post("/v1/responses/compact", [&](const auto &req, auto &res) {
  if (authorized(req, res))
    send(res, failure(501, "This model has no compatible compaction encoder"));
});
auto with_response = [&](const httplib::Request &req, httplib::Response &res, auto action) {
  if (!authorized(req, res))
    return;
  const auto id = req.matches[1].str();
  const auto record = responses.find(id);
  if (!record) {
    send(res, failure(404, "Stored response not found"));
    return;
  }
  try {
    action(id, record);
  } catch (const std::exception &e) {
    send(res, failure(400, e.what()));
  }
};
http.Get(R"(/v1/responses/(resp_[A-Za-z0-9_]+))", [&](const auto &req, auto &res) {
  with_response(req, res, [&](const auto &, const auto &r) {
    for (const auto &[name, value] : req.params)
      server::require(name == "stream" || name == "starting_after",
                      "Unsupported retrieval parameter: " + name);
    const auto stream = req.get_param_value("stream");
    server::require(stream.empty() || stream == "false" || stream == "true", "Invalid stream flag");
    if (stream == "true") {
      server::require(r->stream && r->background,
                      "Stream retrieval requires a streamed background response");
      std::size_t cursor = 0;
      if (req.has_param("starting_after")) {
        const auto after = req.get_param_value("starting_after");
        server::require(!after.empty() && after.size() <= 9 &&
                            after.find_first_not_of("0123456789") == std::string::npos,
                        "Invalid event cursor");
        cursor = std::stoul(after) + 1;
      }
      stream_response(res, r, cursor, nullptr);
    } else {
      server::require(!req.has_param("starting_after"), "starting_after requires stream");
      send(res, {200, r->snapshot()});
    }
  });
});
http.Post(R"(/v1/responses/(resp_[A-Za-z0-9_]+)/cancel)", [&](const auto &req, auto &res) {
  with_response(req, res, [&](const auto &, const auto &r) {
    server::require(r->background, "Only background responses can be cancelled");
    r->cancel_requested = true;
    send(res, {200, r->wait_terminal()});
  });
});
http.Delete(R"(/v1/responses/(resp_[A-Za-z0-9_]+))", [&](const auto &req, auto &res) {
  with_response(req, res, [&](const auto &id, const auto &r) {
    if (!r->finished()) {
      send(res, failure(409, "Cancel an active background response before deleting"));
      return;
    }
    if (!responses.erase(id)) {
      send(res, failure(404, "Stored response not found"));
      return;
    }
    send(res, {200, {{"id", id}, {"object", "response.deleted"}, {"deleted", true}}});
  });
});
http.Get(R"(/v1/responses/(resp_[A-Za-z0-9_]+)/input_items)", [&](const auto &req, auto &res) {
  with_response(req, res, [&](const auto &, const auto &r) {
    for (const auto &[name, value] : req.params)
      server::require(name == "after" || name == "limit" || name == "order",
                      "Unsupported input_items parameter: " + name);
    const auto order = req.has_param("order") ? req.get_param_value("order") : "desc";
    server::require(order == "asc" || order == "desc", "Invalid item order");
    const auto limit = req.has_param("limit") ? req.get_param_value("limit") : "20";
    server::require(!limit.empty() && limit.size() <= 3 &&
                        limit.find_first_not_of("0123456789") == std::string::npos,
                    "Invalid item limit");
    const auto n = std::stoul(limit);
    server::require(n >= 1 && n <= 100, "limit must be 1..100");
    Json items = r->history;
    if (order == "desc")
      std::reverse(items.begin(), items.end());
    auto first = items.begin();
    if (req.has_param("after")) {
      first = std::find_if(first, items.end(), [&](const Json &item) {
        return item.at("id") == req.get_param_value("after");
      });
      server::require(first != items.end(), "Unknown input item cursor");
      ++first;
    }
    Json page = Json::array();
    while (first != items.end() && page.size() < n)
      page.push_back(*first++);
    send(res, {200,
               {{"object", "list"},
                {"data", page},
                {"has_more", first != items.end()},
                {"first_id", page.empty() ? Json(nullptr) : page.front().at("id")},
                {"last_id", page.empty() ? Json(nullptr) : page.back().at("id")}}});
  });
});
