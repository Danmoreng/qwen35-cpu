// Included after Service. Only Service owns inference state; this worker feeds
// independent requests into its existing scheduler and collects completed rows.
class BatchService {
  using Row = std::pair<std::string, Json>;
  struct Active {
    std::size_t index;
    std::shared_ptr<Job> job;
    std::future<Reply> result;
  };
  Service &service_;
  server::BatchDb db_;
  std::size_t context_, width_;
  std::string encoding_;
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false, unavailable_ = false;
  std::thread worker_;
  static constexpr std::size_t output_limit = 32 * 1024 * 1024, storage_limit = 128 * 1024 * 1024;
  void stamp(Json &batch, const std::string &state) {
    batch["status"] = state;
    batch[state + "_at"] = std::time(nullptr);
  }
  void finalize(Json &batch, const std::string &state) {
    const std::string id = batch["id"];
    db_.transaction([&] {
      for (bool ok : {true, false}) {
        auto lines = db_.values("SELECT line FROM results WHERE batch=? AND ok=? ORDER BY idx",
                                {id, ok ? "1" : "0"});
        if (lines.empty())
          continue;
        std::string content;
        for (const auto &line : lines) {
          content += line;
          content += '\n';
        }
        auto info = db_.file(id + (ok ? "-output.jsonl" : "-errors.jsonl"), "batch_output", content);
        batch[ok ? "output_file_id" : "error_file_id"] = info["id"];
      }
      stamp(batch, state);
      db_.save(batch);
      db_.run("DELETE FROM results WHERE batch=?", {id});
    });
  }
  std::shared_ptr<Job> prepare(const std::string &endpoint, const Json &body) {
    server::require(body.is_object() && !body.value("stream", false),
                    "Batch rows require a non-streaming object body");
    if (endpoint == "/v1/systemone") {
      server::require(service_.decisions_enabled(), "This model has no supported qev_config.json");
      auto job = std::make_shared<Job>();
      job->decision = systemone::parse(body.dump(), encoding_);
      return job;
    }
    if (endpoint == "/v1/responses") {
      server::require(!body.value("background", false) && !body.value("store", false),
                      "Batch Responses are stateless; use store=false");
      auto j = body;
      j["store"] = false;
      server::require(!j.contains("previous_response_id") || j["previous_response_id"].is_null(),
                      "Provide full input history for a batch response");
      auto prepared = server::prepare_response(std::move(j), Json::array(), context_, model_name);
      auto job = std::make_shared<Job>();
      static_cast<server::Request &>(*job) = std::move(prepared.request);
      job->truncation_messages = std::move(prepared.formatted_messages);
      job->response_record = std::make_shared<server::ResponseRecord>(prepared);
      return job;
    }
    return parse(body.dump(), context_,
                 endpoint == "/v1/completions" ? server::Api::completion : server::Api::chat);
  }
  std::vector<Row> validate(const Json &batch, const std::string &content, std::size_t &line_no) {
    std::istringstream stream(content);
    std::string line;
    std::vector<Row> rows;
    std::set<std::string> ids;
    while (std::getline(stream, line)) {
      ++line_no;
      server::require(line.size() <= 262144 && line_no <= 4096,
                      "Input exceeds the row size or row count limit");
      auto row = Json::parse(line);
      server::fields(row, {"custom_id", "method", "url", "body"});
      auto id = row.at("custom_id").get<std::string>();
      server::require(!id.empty() && id.size() <= 512 && ids.insert(id).second,
                      "Missing or duplicate custom_id");
      server::require(row.at("method") == "POST" && row.at("url") == batch["endpoint"],
                      "Row method or URL does not match batch endpoint");
      server::require(row.at("body").is_object(), "Expected request body object");
      rows.emplace_back(std::move(id), std::move(row.at("body")));
    }
    server::require(!rows.empty(), "Empty JSONL input");
    return rows;
  }
  void execute(const std::string &id) {
    Json batch;
    std::string content;
    {
      std::lock_guard lock(mutex_);
      batch = db_.batch(id);
      if (batch["status"] == "cancelled")
        return;
      if (batch["status"] == "cancelling") {
        finalize(batch, "cancelled");
        return;
      }
      content = db_.values("SELECT content FROM files WHERE id=?", {batch.at("input_file_id")}).at(0);
    }
    std::size_t line_no = 0;
    std::vector<Row> rows;
    try {
      rows = validate(batch, content, line_no);
    } catch (const std::exception &e) {
      std::lock_guard lock(mutex_);
      batch = db_.batch(id);
      if (batch["status"] == "cancelled")
        return;
      batch["errors"] = {
          {"object", "list"},
          {"data",
           Json::array({{{"code", "invalid_jsonl"}, {"line", line_no}, {"message", e.what()}}})}};
      finalize(batch, batch["status"] == "cancelling" ? "cancelled" : "failed");
      return;
    }
    std::set<std::size_t> done;
    std::size_t written = 0;
    {
      std::lock_guard lock(mutex_);
      batch = db_.batch(id);
      if (batch["status"] == "cancelled")
        return;
      if (batch["status"] == "cancelling") {
        finalize(batch, "cancelled");
        return;
      }
      if (batch["status"] == "validating")
        stamp(batch, "in_progress");
      batch["request_counts"]["total"] = rows.size();
      db_.save(batch);
      for (const auto &idx : db_.values("SELECT idx FROM results WHERE batch=?", {id}))
        done.insert(std::stoull(idx));
      written = std::stoull(
          db_.values("SELECT coalesce(sum(1+length(CAST(line AS BLOB))),0) FROM results WHERE batch=?",
                     {id})
              .front());
    }
    auto record = [&](std::size_t index, Reply reply) {
      const bool ok = reply.status == 200;
      const auto req_id = server::response_id("req_");
      Json line = {
          {"id", server::response_id("batch_req_")},
          {"custom_id", rows[index].first},
          {"error", nullptr},
          {"response",
           {{"request_id", req_id}, {"status_code", reply.status}, {"body", std::move(reply.body)}}}};
      auto data = line.dump();
      if (written + data.size() + 1 > output_limit)
        throw std::runtime_error("Batch result byte limit exceeded");
      std::lock_guard lock(mutex_);
      if (db_.file_bytes() + written + data.size() + 1 > storage_limit)
        throw std::runtime_error("Batch file storage is full");
      auto current = db_.batch(id);
      db_.transaction([&] {
        db_.run("INSERT INTO results VALUES(?,?,?,?)",
                {id, std::to_string(index), ok ? "1" : "0", data});
        auto &counter = current["request_counts"][ok ? "completed" : "failed"];
        counter = counter.get<std::size_t>() + 1;
        db_.save(current);
      });
      written += data.size() + 1;
    };
    std::size_t cursor = 0;
    std::vector<Active> active;
    try {
      for (;;) {
        bool cancel, expired, stop;
        {
          std::lock_guard lock(mutex_);
          batch = db_.batch(id);
          cancel = batch["status"] == "cancelling";
          stop = stopping_;
          expired = std::time(nullptr) >= batch.at("expires_at").get<std::time_t>();
        }
        if (cancel || expired || stop)
          for (auto &a : active)
            a.job->cancelled = true;
        while (!cancel && !expired && !stop && cursor < rows.size() && active.size() < width_) {
          if (done.contains(cursor)) {
            ++cursor;
            continue;
          }
          std::shared_ptr<Job> job;
          try {
            job = prepare(batch.at("endpoint"), rows[cursor].second);
          } catch (const std::exception &e) {
            record(cursor++, failure(400, e.what()));
            continue;
          }
          auto result = job->promise.get_future();
          if (!service_.enqueue(job))
            break;
          active.push_back({cursor++, std::move(job), std::move(result)});
        }
        for (auto it = active.begin(); it != active.end();) {
          if (it->result.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            ++it;
            continue;
          }
          auto reply = it->result.get();
          if (reply.status == 200 && it->job->response_record)
            reply.body = it->job->response_record->snapshot();
          if (reply.status != 499 || !(cancel || expired || stop))
            record(it->index, std::move(reply));
          it = active.erase(it);
        }
        if (active.empty() && (cursor == rows.size() || cancel || expired || stop)) {
          std::lock_guard lock(mutex_);
          if (stop)
            return; // durable jobs resume unfinished indices on restart
          batch = db_.batch(id);
          finalize(batch, batch["status"] == "cancelling" ? "cancelled"
                          : expired                       ? "expired"
                                                          : "completed");
          return;
        }
        std::unique_lock lock(mutex_);
        wake_.wait_for(lock, std::chrono::milliseconds(2));
      }
    } catch (...) {
      for (auto &a : active)
        a.job->cancelled = true;
      throw;
    }
  }
  void run() {
    for (;;) {
      std::unique_lock lock(mutex_);
      if (stopping_)
        return;
      auto jobs = db_.values("SELECT id FROM batches WHERE json_extract(info,'$.status') IN "
                             "('validating','in_progress','cancelling') ORDER BY rowid LIMIT 1");
      if (jobs.empty()) {
        wake_.wait(lock);
        continue;
      }
      auto id = jobs.front();
      lock.unlock();
      try {
        execute(id);
      } catch (const std::exception &e) {
        std::lock_guard guard(mutex_);
        auto batch = db_.batch(id);
        batch["errors"] = {{"object", "list"},
                           {"data", Json::array({{{"code", "batch_failed"}, {"message", e.what()}}})}};
        finalize(batch, "failed");
      }
    }
  }

public:
  BatchService(Service &service, const std::string &path, std::size_t context, std::size_t width,
               std::string encoding)
      : service_(service), db_(path), context_(context), width_(width), encoding_(std::move(encoding)),
        worker_([this] {
          try {
            run();
          } catch (const std::exception &e) {
            std::lock_guard lock(mutex_);
            unavailable_ = true;
            std::cerr << "Batch worker stopped: " << e.what() << '\n';
          }
        }) {}
  ~BatchService() {
    {
      std::lock_guard guard(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    worker_.join();
  }
  Reply upload(const std::string &name, const std::string &content) {
    if (name.empty() || name.size() > 255 || content.size() > 4 * 1024 * 1024)
      return failure(400, "Invalid batch filename or file exceeds 4 MiB");
    std::lock_guard lock(mutex_);
    if (db_.file_bytes() + content.size() > storage_limit - output_limit ||
        std::stoull(db_.values("SELECT count(*) FROM files").front()) >= 4096)
      return failure(429, "Batch file storage is full; delete old files");
    return {200, db_.file(name, "batch", content)};
  }
  Reply create(const Json &j) {
    server::fields(j, {"input_file_id", "endpoint", "completion_window", "metadata"});
    const auto endpoint = j.at("endpoint").get<std::string>();
    server::require(endpoint == "/v1/chat/completions" || endpoint == "/v1/responses" ||
                        endpoint == "/v1/completions" || endpoint == "/v1/systemone",
                    "Unsupported batch endpoint");
    server::require(j.at("completion_window") == "24h", "completion_window must be 24h");
    auto meta = j.value("metadata", Json(nullptr));
    server::require(meta.is_null() || (meta.is_object() && meta.size() <= 16), "Invalid metadata");
    for (const auto &[k, v] : meta.items())
      server::require(k.size() <= 64 && v.is_string() && v.get_ref<const std::string &>().size() <= 512,
                      "Invalid metadata entry");
    std::lock_guard lock(mutex_);
    auto file = db_.file_info(j.at("input_file_id"));
    if (file.is_null())
      return failure(404, "Input file not found");
    server::require(file["purpose"] == "batch", "Expected a batch input file");
    auto now = std::time(nullptr);
    Json b = {{"object", "batch"},
              {"id", server::response_id("batch_")},
              {"endpoint", endpoint},
              {"input_file_id", file["id"]},
              {"completion_window", "24h"},
              {"status", "validating"},
              {"created_at", now},
              {"expires_at", now + 86400},
              {"metadata", meta},
              {"errors", nullptr},
              {"output_file_id", nullptr},
              {"error_file_id", nullptr},
              {"request_counts", {{"total", 0}, {"completed", 0}, {"failed", 0}}}};
    for (auto key : {"in_progress_at", "finalizing_at", "completed_at", "failed_at", "cancelling_at",
                     "cancelled_at", "expired_at"})
      b[key] = nullptr;
    if (unavailable_)
      return failure(503, "Batch worker unavailable; check server log and database storage");
    if (std::stoull(db_.values("SELECT count(*) FROM batches WHERE json_extract(info,'$.status') IN "
                               "('validating','in_progress','cancelling')")
                        .front()) >= 64)
      return failure(429, "Batch queue is full");
    db_.run("INSERT INTO batches VALUES(?,?)", {b["id"], b.dump()});
    wake_.notify_one();
    return {200, b};
  }
  Reply object(const std::string &id, bool file, bool cancel = false) {
    std::lock_guard lock(mutex_);
    auto info = file ? db_.file_info(id) : db_.batch(id);
    if (info.is_null())
      return failure(404, "Object not found");
    if (cancel && info["status"] == "validating") {
      stamp(info, "cancelled");
      db_.save(info);
      wake_.notify_one();
    } else if (cancel && info["status"] == "in_progress") {
      stamp(info, "cancelling");
      db_.save(info);
      wake_.notify_one();
    }
    return {200, info};
  }
  std::optional<std::string> content(const std::string &id) {
    std::lock_guard lock(mutex_);
    auto rows = db_.values("SELECT content FROM files WHERE id=?", {id});
    if (rows.empty())
      return {};
    return std::move(rows.front());
  }
  Reply remove_file(const std::string &id) {
    std::lock_guard lock(mutex_);
    if (db_.file_info(id).is_null())
      return failure(404, "File not found");
    if (!db_.values("SELECT id FROM batches WHERE json_extract(info,'$.input_file_id')=? AND "
                    "json_extract(info,'$.status') IN ('validating','in_progress','cancelling') LIMIT 1",
                    {id})
             .empty())
      return failure(409, "Active batch still uses this file");
    db_.run("DELETE FROM files WHERE id=?", {id});
    return {200, {{"id", id}, {"object", "file"}, {"deleted", true}}};
  }
  Reply list(bool files, const httplib::Request &req) {
    auto limit = req.get_param_value("limit", std::size_t(0));
    if (limit.empty())
      limit = "20";
    server::require(limit.size() <= 3 && limit.find_first_not_of("0123456789") == std::string::npos &&
                        std::stoi(limit) >= 1 && std::stoi(limit) <= 100,
                    "limit must be 1..100");
    auto order = req.has_param("order") ? req.get_param_value("order") : "desc";
    server::require(order == "desc" || (files && order == "asc"), "Invalid list order");
    const std::string table = files ? "files" : "batches";
    std::vector<std::string> args;
    std::string sql = "SELECT info FROM " + table + " WHERE 1";
    std::lock_guard lock(mutex_);
    if (req.has_param("after")) {
      const auto after = req.get_param_value("after");
      auto row = db_.values(("SELECT rowid FROM " + table + " WHERE id=?").c_str(), {after});
      if (row.empty())
        return failure(400, "Unknown pagination cursor");
      sql += order == "asc" ? " AND rowid>?" : " AND rowid<?";
      args.push_back(row.front());
    }
    if (files && req.has_param("purpose")) {
      sql += " AND json_extract(info,'$.purpose')=?";
      args.push_back(req.get_param_value("purpose"));
    }
    sql += order == "asc" ? " ORDER BY rowid ASC LIMIT ?" : " ORDER BY rowid DESC LIMIT ?";
    std::size_t n = std::stoul(limit);
    args.push_back(std::to_string(n + 1));
    auto rows = db_.values(sql.c_str(), args);
    Json data = Json::array();
    for (std::size_t i = 0; i < std::min(n, rows.size()); ++i)
      data.push_back(Json::parse(rows[i]));
    return {200,
            {{"object", "list"},
             {"data", data},
             {"first_id", data.empty() ? Json(nullptr) : data.front()["id"]},
             {"last_id", data.empty() ? Json(nullptr) : data.back()["id"]},
             {"has_more", rows.size() > n}}};
  }
};
