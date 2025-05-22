#pragma once

#include <cassert>
#include <vector>
#include <stdexcept>

#include <dlib/serialize.h>


namespace huffman {
    template <typename T>
    class decoder {
    private:
        class node {
        public:
            using value_type = T;
            using nodes_type = std::vector<node>;
            
        private:
            /* invariant:
             *   bits  < 0  value present
             *   bits == 0  end-of-stream marker
             *   bits  > 0  2^bits children
             */
            int bits;
            unsigned mask;  // (1<<bits)-1 when bits > 0
            union {
                value_type value;
                nodes_type children;
            };

            void destruct() {
                if (bits < 0)
                    value.~value_type();
                else if (bits > 0)
                    children.~nodes_type();
            }

            static int log2(const nodes_type& nodes) {
                int r = 0;
                for (auto n = nodes.size()>>1; n > 0; n>>=1) ++r;
                return r;
            }

            
        public:
            ~node() { destruct(); }

            node() : bits(0) {}  // end-of-stream marker
            node(T value) noexcept : bits(-1), value(std::move(value)) {}
            node(nodes_type&& children) noexcept
                : bits(log2(children)),
                  mask((1u<<bits)-1),
                  children(move(children)) {
                assert(bits > 0 && this->children.size() == (1u<<bits));
            }

            node(node&& other) noexcept : bits(other.bits), mask(other.mask) {
                if (bits < 0)
                    new (&value) value_type(std::move(other.value));
                else if (bits > 0)
                    new (&children) nodes_type(move(other.children));
            }
            node& operator=(node&& other) {
                destruct();
                bits = other.bits;
                mask = other.mask;
                if (bits < 0)
                    new (&value) value_type(std::move(other.value));
                else if (bits > 0)
                    new (&children) nodes_type(move(other.children));
                return *this;
            }

            node(const node&) = delete;
            node& operator=(const node&) = delete;
                

            inline unsigned bits_needed() const {
                return bits > 0 ? unsigned(bits) : 0;
            }

            inline const node* next(unsigned i) const {
                return children.data() + (i&mask);
            }

            inline const T* value_ptr() const {
                return bits < 0 ? &value : nullptr;
            }

            unsigned min_depth() const {
                if (bits <= 0) return 0;
                unsigned r = 256;
                for (auto& n : children) {
                    const auto d = n.min_depth();
                    if (r > d)
                        r = d;
                }
                return unsigned(bits) + r;
            }

            template <typename OutIter>
            void move_children(OutIter&& out, unsigned depth) {
                assert(bits == 1);
                if (depth > 0) {
                    --depth;
                    for (auto& n : children)
                        n.move_children(out, depth);
                }
                else
                    for (auto& n : children) {
                        *out = std::move(n);
                        ++out;
                    }
            }
            
            void flatten() {
                if (bits != 1) return;
                unsigned r = 7;
                for (auto& n : children) {
                    const auto d = n.min_depth();
                    if (r > d)
                        r = d;
                }
                if (r > 0) {
                    const auto new_bits = bits + int(r);
                    nodes_type new_children;
                    new_children.reserve(1u<<new_bits);
                    move_children(back_inserter(new_children), r);

                    children = move(new_children);
                    bits = new_bits;
                    mask = (1u<<bits)-1;
                    assert(children.size() == (1u<<bits));
                }
                for (auto& n : children)
                    n.flatten();  // recurse
            }
            
            void serialize(std::ostream& out) const {
                if (bits < 0) {
                    out << 'v';
                    dlib::serialize(value, out);
                }
                else if (bits == 0)
                    out << 'e';
                else if (bits == 1) {
                    out << 'b';
                    for (auto& node : children)
                        node.serialize(out);
                }
                else {  // bits > 1
                    out << char('0'+bits);
                    for (auto& node : children)
                        node.serialize(out);
                }
            }
        };

        node root;

        unsigned buf = 0;
        unsigned nvalid = 0;

        decoder(decoder&&) = default;
        decoder& operator=(decoder&&) = default;

        decoder(const decoder&) = delete;
        decoder& operator=(const decoder&) = delete;

        
        static node deserialize_node(std::istream& in) {
            char c;
            in >> c;
            switch (c) {
            case 'v': {
                T value;
                dlib::deserialize(value, in);
                return { std::move(value) };
            }
                
            case '0':
            case 'e':
                return {}; // end-of-stream marker
                
            case '1':
            case 'b': {
                typename node::nodes_type children;
                children.reserve(2);
                children.emplace_back(deserialize_node(in));
                children.emplace_back(deserialize_node(in));
                return { move(children) };
            }
                
            default:
                if (c >= '2' && c <= '8') {
                    typename node::nodes_type children;
                    auto n = 1u << (c-'0');
                    children.reserve(n);
                    for ( ; n > 0; --n)
                        children.emplace_back(deserialize_node(in));
                    return { move(children) };
                }
                throw dlib::serialization_error("Invalid node found while deserializing huffman::decoder.");
            }
        }

        template <typename NODE>
        static node init(const NODE& n) {
            if (n.n0) {
                typename node::nodes_type children;
                children.emplace_back(init(*n.n0));
                children.emplace_back(init(*n.n1));
                return { move(children) };
            }
            else if (n.value)
                return { *n.value };
            else
                return {}; // end-of-stream marker
        }
        

    public:
        decoder() = default;

        template <typename NODE>
        explicit decoder(const NODE& root)
            : root(init(root)) {
            this->root.flatten();
        }

        inline bool empty() const {
            return root.bits_needed() == 0;
        }
        
        // returns nullptr at end of stream
        const T* operator()(std::istream& in) {
            const auto* node = &root;
            while (auto n = node->bits_needed()) {
                if (nvalid < n) {
                    char c;
                    in.get(c);
                    if (!in.good())
                        throw std::runtime_error("huffman read failed");
                    buf <<= 8;
                    buf += unsigned(c) & 0xff;
                    nvalid += 8;
                }
                nvalid -= n;
                node = node->next(buf>>nvalid);
            }
            if (auto p = node->value_ptr())
                return p;
            // end-of-stream: reset decoder
            if (buf & ~((~0u)<<nvalid))
                throw std::runtime_error("huffman stream corrupt");
            nvalid = 0;
            return nullptr;
        }
        
        void deserialize(std::istream& in) {
            int version = 0;
            dlib::deserialize(version, in);
            if (version != 1)
                throw dlib::serialization_error("Unexpected version found while deserializing huffman::decoder.");
            root = deserialize_node(in);
            root.flatten();
        }
        friend void deserialize(decoder& item, std::istream& in) {
            item.deserialize(in);
        }

        void serialize(std::ostream& out) const {
            static constexpr auto version = 1;
            dlib::serialize(version, out);
            root.serialize(out);
        }
        friend void serialize(const decoder& item, std::ostream& out) {
            item.serialize(out);
        }
    };
}
