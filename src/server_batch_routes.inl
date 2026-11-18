auto batch_route = [&](const httplib::Request &req, httplib::Response &res, auto action) {
  if (!authorized(req, res))
    return;
  try {
    send(res, action());
  } catch (const server::BatchStorageError &e) {
    send(res, failure(500, e.what()));
  } catch (const std::exception &e) {
    send(res, failure(400, e.what()));
  }
};
http.Post("/v1/files", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] {
    server::require(req.is_multipart_form_data() && req.form.get_field_count("purpose") == 1 &&
                        req.form.get_field("purpose") == "batch" && req.form.get_file_count("file") == 1,
                    "Use multipart file and purpose=batch");
    auto f = req.form.get_file("file");
    return batches.upload(f.filename, f.content);
  });
});
http.Get("/v1/files", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.list(true, req); });
});
http.Get(R"(/v1/files/([^/]+))", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.object(req.matches[1], true); });
});
http.Delete(R"(/v1/files/([^/]+))", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.remove_file(req.matches[1]); });
});
http.Get(R"(/v1/files/([^/]+)/content)", [&](const auto &req, auto &res) {
  if (!authorized(req, res))
    return;
  const auto data = batches.content(req.matches[1]);
  if (!data)
    send(res, failure(404, "File not found"));
  else
    res.set_content(*data, "application/jsonl");
});
http.Post("/v1/batches", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.create(Json::parse(req.body)); });
});
http.Get("/v1/batches", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.list(false, req); });
});
http.Get(R"(/v1/batches/([^/]+))", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.object(req.matches[1], false); });
});
http.Post(R"(/v1/batches/([^/]+)/cancel)", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return batches.object(req.matches[1], false, true); });
});
http.Get("/v1/batch-config", [&](const auto &req, auto &res) {
  batch_route(req, res, [&] { return Reply{200, service.batch_tuning().report()}; });
});
