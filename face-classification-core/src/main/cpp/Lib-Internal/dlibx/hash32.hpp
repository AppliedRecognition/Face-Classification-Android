#pragma once

#include <string>
#include <string_view>

namespace dlibx {
    /** \brief Compute 7 character base32 hash of string.
     *
     * Different ids give different hashes.
     *
     * The least significant 3 bits of id appear in the first character
     * of the output.  
     * If (id%8) < 6, then the first character is a letter.
     * If (id%8) = 6, then the first character may be a letter or a number.
     * If (id%8) = 7, then the first character is a number.
     * It is possible to recover those 3 bits from the output.
     */
    std::string hash32(std::string_view str, unsigned id = 0);
}
