#pragma once

#include "containers.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// void to_json(json &j, int3 const &i);
// void from_json(json const &j, int3 &i);
// void to_json(json &j, dim3 const &i);
// void from_json(json const &j, dim3 &i);

// inline void to_json(json &j, int3 const &i) {
//   j["x"] = i.x;
//   j["y"] = i.y;
//   j["z"] = i.z;
// }

// inline void from_json(json const &j, int3 &i) {
//   j.at("x").get_to(i.x);
//   j.at("y").get_to(i.y);
//   j.at("z").get_to(i.z);
// }

// inline void to_json(json &j, dim3 const &i) {
//   j["x"] = i.x;
//   j["y"] = i.y;
//   j["z"] = i.z;
// }

// inline void from_json(json const &j, dim3 &i) {
//   j.at("x").get_to(i.x);
//   j.at("y").get_to(i.y);
//   j.at("z").get_to(i.z);
// }

template <typename T>
T load_json(char const *file_path) {
  std::ifstream ifs(file_path);
  json j;
  ifs >> j;
  return j.get<T>();
}