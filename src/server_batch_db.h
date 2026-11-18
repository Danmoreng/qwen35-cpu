#pragma once
#include "server_responses.h"
#include <sqlite3.h>
#include <sstream>

namespace qwen35x::server {
// The dispatcher serializes all database access. Statements own their bindings.
class BatchStorageError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
class BatchDb {
  sqlite3 *db_ = nullptr;
  void check(int rc) const {
    if (rc != SQLITE_OK)
      throw BatchStorageError(std::string("Batch database: ") + sqlite3_errmsg(db_));
  }
  struct Statement {
    sqlite3_stmt *p = nullptr;
    ~Statement() { sqlite3_finalize(p); }
  };
  void prepare(Statement &s, const char *sql, const std::vector<std::string> &args) {
    check(sqlite3_prepare_v2(db_, sql, -1, &s.p, nullptr));
    for (std::size_t i = 0; i < args.size(); ++i)
      check(sqlite3_bind_text(s.p, int(i + 1), args[i].data(), int(args[i].size()), SQLITE_TRANSIENT));
  }

public:
  explicit BatchDb(const std::string &path) {
    int rc = sqlite3_open(path.c_str(), &db_);
    try {
      check(rc);
      run("PRAGMA locking_mode=EXCLUSIVE");
      run("BEGIN EXCLUSIVE");
      run("CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY, info TEXT NOT NULL, content TEXT NOT "
          "NULL)");
      run("CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY, info TEXT NOT NULL)");
      run("CREATE TABLE IF NOT EXISTS results(batch TEXT, idx INTEGER, ok INTEGER NOT NULL, line TEXT "
          "NOT NULL, PRIMARY KEY(batch,idx))");
      run("COMMIT");
    } catch (...) {
      sqlite3_close(db_);
      throw;
    }
  }
  BatchDb(const BatchDb &) = delete;
  ~BatchDb() { sqlite3_close(db_); }
  void run(const char *sql, const std::vector<std::string> &args = {}) {
    Statement s;
    prepare(s, sql, args);
    int rc;
    do {
      rc = sqlite3_step(s.p);
    } while (rc == SQLITE_ROW);
    if (rc != SQLITE_DONE)
      check(rc);
  }
  std::vector<std::string> values(const char *sql, const std::vector<std::string> &args = {}) {
    Statement s;
    prepare(s, sql, args);
    std::vector<std::string> out;
    int rc;
    while ((rc = sqlite3_step(s.p)) == SQLITE_ROW) {
      const auto *p = sqlite3_column_text(s.p, 0);
      out.emplace_back(p ? reinterpret_cast<const char *>(p) : "", sqlite3_column_bytes(s.p, 0));
    }
    if (rc != SQLITE_DONE)
      check(rc);
    return out;
  }
  template <class F> void transaction(F fn) {
    run("BEGIN");
    try {
      fn();
      run("COMMIT");
    } catch (...) {
      run("ROLLBACK");
      throw;
    }
  }
  Json file(const std::string &name, const std::string &purpose, const std::string &content) {
    Json info = {{"object", "file"},        {"id", response_id("file-")},
                 {"filename", name},        {"purpose", purpose},
                 {"bytes", content.size()}, {"created_at", std::time(nullptr)},
                 {"status", "processed"}};
    run("INSERT INTO files VALUES(?,?,?)", {info["id"], info.dump(), content});
    return info;
  }
  std::size_t file_bytes() {
    return std::stoull(
        values("SELECT coalesce(sum(json_extract(info,'$.bytes')),0) FROM files").front());
  }
  void save(const Json &batch) {
    run("UPDATE batches SET info=? WHERE id=?", {batch.dump(), batch.at("id")});
  }
  Json get(const char *sql, const std::string &id) {
    const auto rows = values(sql, {id});
    return rows.empty() ? Json(nullptr) : Json::parse(rows.front());
  }
  Json batch(const std::string &id) { return get("SELECT info FROM batches WHERE id=?", id); }
  Json file_info(const std::string &id) { return get("SELECT info FROM files WHERE id=?", id); }
};
} // namespace qwen35x::server
