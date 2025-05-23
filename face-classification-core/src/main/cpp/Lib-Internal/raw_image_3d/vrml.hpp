#pragma once

#include <ostream>
#include <string_view>
#include <variant>
#include <vector>

namespace vrml {
    struct value;
    using string = std::string;
    using array = std::vector<value>;
    struct object {
        std::string class_name;
        std::vector<std::pair<string, value> > members;
    };
    struct value : std::variant<string,array,object> {
        using variant = std::variant<string,array,object>;
        using variant::variant;
    };

    std::ostream& render(std::ostream& out, const array& arr,
                         const std::string& indent = "");
    std::ostream& render(std::ostream& out, const object& obj,
                         const std::string& indent = "");
    std::ostream& render(std::ostream& out, const value& v,
                         const std::string& indent = "");

    inline std::ostream& operator<<(std::ostream& out, const value& v) {
        return render(out,v);
    }

    // returns true if str is not empty, false if nothing left
    bool consume_whitespace_and_comments(std::string_view& str);

    // upon return the string_view will only contain what follows the vrml
    value parse(std::string_view& str);
}

