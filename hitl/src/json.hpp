// json.hpp — minimal dependency-free JSON reader.
//
// Only what the scene loader needs: objects, arrays, numbers, strings, bools,
// null. No writer, no unicode escapes beyond \uXXXX → UTF-8 for the BMP.
#pragma once

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mjson {

struct Value;
using Object = std::map<std::string, Value>;
using Array  = std::vector<Value>;

struct Value {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ };

    Type        type = NUL;
    bool        b    = false;
    double      num  = 0.0;
    std::string str;
    Array       arr;
    Object      obj;

    bool has(const std::string& k) const {
        return type == OBJ && obj.find(k) != obj.end();
    }

    const Value& at(const std::string& k) const {
        auto it = obj.find(k);
        if (type != OBJ || it == obj.end())
            throw std::runtime_error("json: missing key '" + k + "'");
        return it->second;
    }

    double number() const {
        if (type != NUM) throw std::runtime_error("json: expected a number");
        return num;
    }

    double number_or(const std::string& k, double fallback) const {
        return has(k) ? at(k).number() : fallback;
    }

    bool bool_or(const std::string& k, bool fallback) const {
        if (!has(k)) return fallback;
        const Value& v = at(k);
        if (v.type != BOOL) throw std::runtime_error("json: expected a bool for '" + k + "'");
        return v.b;
    }

    std::string string_or(const std::string& k, const std::string& fallback) const {
        if (!has(k)) return fallback;
        const Value& v = at(k);
        if (v.type != STR) throw std::runtime_error("json: expected a string for '" + k + "'");
        return v.str;
    }

    // Fixed-length float vector, e.g. "c": [x, y, z].
    void floats(float* out, size_t n, const char* what) const {
        if (type != ARR || arr.size() != n)
            throw std::runtime_error(std::string("json: ") + what + " must be an array of "
                                     + std::to_string(n) + " numbers");
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(arr[i].number());
    }
};

namespace detail {

struct Parser {
    const char* p;
    const char* end;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("json: " + msg);
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    char peek() {
        skip_ws();
        if (p >= end) fail("unexpected end of input");
        return *p;
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++p;
    }

    bool literal(const char* lit) {
        size_t n = std::char_traits<char>::length(lit);
        if (static_cast<size_t>(end - p) < n) return false;
        if (std::string(p, n) != lit) return false;
        p += n;
        return true;
    }

    std::string parse_string() {
        expect('"');
        std::string s;
        while (p < end && *p != '"') {
            if (*p != '\\') { s += *p++; continue; }
            ++p;
            if (p >= end) fail("unterminated escape");
            switch (*p++) {
                case '"':  s += '"';  break;
                case '\\': s += '\\'; break;
                case '/':  s += '/';  break;
                case 'b':  s += '\b'; break;
                case 'f':  s += '\f'; break;
                case 'n':  s += '\n'; break;
                case 'r':  s += '\r'; break;
                case 't':  s += '\t'; break;
                case 'u': {
                    if (end - p < 4) fail("short \\u escape");
                    unsigned cp = static_cast<unsigned>(std::strtoul(std::string(p, 4).c_str(), nullptr, 16));
                    p += 4;
                    if (cp < 0x80) {
                        s += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        s += static_cast<char>(0xC0 | (cp >> 6));
                        s += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        s += static_cast<char>(0xE0 | (cp >> 12));
                        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        s += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;  // closing quote
        return s;
    }

    Value parse_value() {
        char c = peek();
        Value v;
        if (c == '{') {
            ++p;
            v.type = Value::OBJ;
            if (peek() == '}') { ++p; return v; }
            for (;;) {
                std::string key = parse_string();
                expect(':');
                v.obj.emplace(std::move(key), parse_value());
                char d = peek();
                ++p;
                if (d == '}') break;
                if (d != ',') fail("expected ',' or '}' in object");
            }
        } else if (c == '[') {
            ++p;
            v.type = Value::ARR;
            if (peek() == ']') { ++p; return v; }
            for (;;) {
                v.arr.push_back(parse_value());
                char d = peek();
                ++p;
                if (d == ']') break;
                if (d != ',') fail("expected ',' or ']' in array");
            }
        } else if (c == '"') {
            v.type = Value::STR;
            v.str  = parse_string();
        } else if (literal("true")) {
            v.type = Value::BOOL; v.b = true;
        } else if (literal("false")) {
            v.type = Value::BOOL; v.b = false;
        } else if (literal("null")) {
            v.type = Value::NUL;
        } else {
            char* tail = nullptr;
            double d = std::strtod(p, &tail);
            if (tail == p) fail("bad value");
            p = tail;
            v.type = Value::NUM;
            v.num  = d;
        }
        return v;
    }
};

}  // namespace detail

inline Value parse(const std::string& text) {
    detail::Parser ps{text.data(), text.data() + text.size()};
    Value v = ps.parse_value();
    ps.skip_ws();
    if (ps.p != ps.end) throw std::runtime_error("json: trailing content after value");
    return v;
}

}  // namespace mjson
