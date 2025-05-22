
#include "vrml.hpp"

#include <applog/core.hpp>

std::ostream&
vrml::render(std::ostream& out, const array& arr, const std::string& indent) {
    for (auto& v : arr)
        if (!std::holds_alternative<string>(v)) {
            // array holds other array or object so use indenting
            out << '[' << std::endl;
            const auto next = indent + '\t';
            for (auto& v : arr) {
                out << next;
                render(out, v, next);
                out << std::endl; // no commas
            }
            return out << indent << ']';
        }

    // simple array of string -- all on one line
    if (arr.empty())
        out << '[';
    else {
        char comma = '[';
        for (auto& v : arr) {
            out << comma;
            render(out, v);
            comma = ',';
        }
    }
    return out << ']';
}

std::ostream&
vrml::render(std::ostream& out, const object& obj, const std::string& indent) {
    out << obj.class_name << " {" << std::endl;
    const auto next = indent + '\t';
    for (auto& pr : obj.members) {
        out << next << pr.first << ' ';
        render(out, pr.second, next);
        out << std::endl;
    }
    return out << indent << '}';
}

std::ostream&
vrml::render(std::ostream& out, const value& v, const std::string& indent) {
    if (auto* p = std::get_if<string>(&v))
        return out << '"' << *p << '"';
    if (auto* p = std::get_if<array>(&v))
        return render(out, *p, indent);
    if (auto* p = std::get_if<object>(&v))
        return render(out, *p, indent);
    return out << "[?VRML?]"; // shouldn't happen
}

bool vrml::consume_whitespace_and_comments(std::string_view& str) {
    while (!str.empty()) {
        if (isspace(str.front()))
            str.remove_prefix(1);
        else if (str.front() != '#')
            return true;
        else // comment
            do {
                str.remove_prefix(1);
            } while (!str.empty() && str.front() != '\n');
    }
    return false;
}

static auto parse_array(std::string_view& str) {
    vrml::array arr;
    if (!vrml::consume_whitespace_and_comments(str))
        throw std::runtime_error("VRML array '[' not found");
    str.remove_prefix(1);

    // elements
    for (;;) {
        if (!vrml::consume_whitespace_and_comments(str))
            throw std::runtime_error("VRML array element or ']' not found");
        if (str.front() == ']') {
            str.remove_prefix(1);
            break; // done
        }
        if (str.front() == ',') {
            str.remove_prefix(1);
            continue;
        }
        arr.emplace_back(vrml::parse(str));
    }
    
    return arr;
}

static auto parse_object(std::string_view& str) {
    vrml::object obj;

    // parse class name
    for (;;) {
        if (!vrml::consume_whitespace_and_comments(str))
            throw std::runtime_error("VRML object class name not found");

        if ('A' <= str.front() && str.front() <= 'Z') {
            // class name
            const auto open = str.find('{');
            if (str.size() <= open) {
                FILE_LOG(logERROR) << "VRML: '" << str.substr(0,20) << "'";
                throw std::runtime_error("VRML object '{' not found");
            }
            auto name = str.substr(0,open);
            while (isspace(name.back()))
                name.remove_suffix(1);
            for (auto c : name)
                if (!isalnum(c)) {
                    FILE_LOG(logERROR) << "VRML: '" << name << "'";
                    throw std::runtime_error("VRML class name invalid");
                }
            obj.class_name = name;
            str.remove_prefix(open+1);
            break;
        }

        FILE_LOG(logERROR) << "VRML: '" << str.substr(0,20) << "'";
        throw std::runtime_error("VRML class name not found");
    }

    // parse object members
    for (;;) {
        if (!vrml::consume_whitespace_and_comments(str))
            throw std::runtime_error("VRML object member or '}' not found");

        if (str.front() == '}') {
            str.remove_prefix(1);
            break; // done
        }
        
        if ('a' <= str.front() && str.front() <= 'z') {
            auto sep = str.find_first_of(" \f\n\r\t\v");
            if (str.size() <= sep) {
                FILE_LOG(logERROR) << "VRML: '" << str.substr(0,20) << "'";
                throw std::runtime_error("VRML object member not found");
            }
            auto name = str.substr(0,sep);
            for (auto c : name)
                if (!isalnum(c)) {
                    FILE_LOG(logERROR) << "VRML: '" << name << "'";
                    throw std::runtime_error("VRML member name invalid");
                }
            str.remove_prefix(sep+1);
            obj.members.emplace_back(name, vrml::parse(str));
            continue;
        }

        FILE_LOG(logERROR) << "VRML: '" << str.substr(0,20) << "'";
        throw std::runtime_error("VRML member name or '}' not found");
    }
    
    return obj;
}

vrml::value vrml::parse(std::string_view& str) {
    if (!consume_whitespace_and_comments(str))
        throw std::runtime_error("VRML value not found");

    if (str.front() == '[')
        return parse_array(str);

    if (str.front() == '"') {
        // quoted string
        str.remove_prefix(1);
        auto end = str.find('"');
        if (str.size() <= end) {
            FILE_LOG(logERROR) << "VRML: '" << str.substr(0,20) << "'";
            throw std::runtime_error("VRML end of string '\"' not found");
        }
        auto s = str.substr(0,end);
        str.remove_prefix(end+1);
        return string(s);
    }

    if ('A' <= str.front() && str.front() <= 'Z') { // object?
        for (auto c : str) {
            if (isalnum(c) || c == ' ')
                continue;
            if (c == '{')
                return parse_object(str);
            break;
        }
    }

    // else unquoted string to end of line, comma, ] or }
    auto end = str.find_first_of(",]}\n");
    if (str.size() <= end) {
        FILE_LOG(logERROR) << "VRML: '" << str.substr(0,20) << "'";
        throw std::runtime_error("VRML end of string not found");
    }
    auto s = str.substr(0,end);
    str.remove_prefix(end);
    while (isspace(s.back()))
        s.remove_suffix(1);
    return string(s);
}
