#pragma once

#include <ostream>
#include <memory>

#include <optional>

#include "types.hpp"


namespace json {


    namespace detail {

        /** \brief Puller type stream for container.
         *
         * The internal state of the stream includes the following: <ul>
         *   <li>an buffer of values</li>
         *   <li>a handler from which to pull values</li>
         *   <li>an optional final_size</li>
         *   <li>a final flag</li>
         * </ul>
         *
         * Values in the buffer will be pulled out before values returned
         * by the handler. 
         *
         * The handler may return the equivalent of nullopt to indicate
         * end-of-stream (setting the final flag).
         *
         * The final_size may be specified in advance or will be determined
         * once the end-of-stream signal is seen.
         *
         * The final flag indicates that the stream is coming to an end,
         * not that it has ended.  
         * Values may still be pulled after the final flag has been set.
         * The final size will, however, always be known once the final flag
         * is set.
         */
        template <typename T, typename FINAL, class OPS>
        class basic_puller {
        public:
            /** \brief Streamed element type.
             */
            using value_type = T;

            /** \brief Pull type.
             */
            using result_type = std::optional<T>;

            /** \brief Type for final container of elements.
             */
            using final_type = FINAL;

            /** \brief Integer size type.
             */
            using size_type = std::size_t;


            /** \brief Set final size.
             *
             * This method may not be called if a final size has already been
             * set, the final flag is set, or if more than the specified size
             * has already passed through.
             * When complete, the total data streamed must equal this value.
             *
             * \param[in] final_size final size specification
             */
            void set_final_size(size_type final_size);


            /** \brief Final stream size (if known).
             *
             * If the final size is not yet known, nullopt is returned.
             * When the end-of-stream is reached, this value is set to the
             * total data streamed.
             *
             * The metric used to tally the final size is type dependent.
             * For some types (array and object) this will be the number of
             * elements passed while for other types (string and binary) it
             * will be the number of characters or bytes passed.
             *
             * \return final size or nullopt if unknown
             */
            const std::optional<size_type>& final_size() const;


            /** \brief Set handler to pull data from.
             *
             * The handler must take no arguments and is expected to return
             * a value convertable to std::optional<value_type>.  
             * The returned value must be the equivalent of nullopt
             * to indicate end-of-stream.
             *
             * The handler may be copied when set, but will not be copied
             * again once it starts being called.
             *
             * \pre no handler has been set and the stream is not in the final state 
             * \param[in] handler pull handler
             */
            template <typename HANDLER>
            inline void set_handler(const HANDLER& handler) {
                set_handler_obj(
                    std::make_unique<handler_obj<HANDLER> >(handler));
            }

            
            /** \brief Pull element from stream.
             *
             * If the internal buffer is non-empty, the value at the head of
             * the buffer is removed and returned.
             * If the internal buffer is empty and a handler is set, the
             * handler is called to obtain a value to return.
             * If the handler is not set or it does not return value, 
             * the the final flag is set and nullopt is returned.
             *
             * Any exceptions thrown by the handler are propagated out without
             * altering the internal state of the stream.
             *
             * When the final flag is set, the final_size will also be set.
             * Note that the final flag (and final_size) may be set before
             * the end-of-stream is reached.
             *
             * \return element or nullopt to indicate end of stream
             */
            result_type operator()();


            /** \brief Push value into internal buffer.
             *
             * If the argument is nullopt, the stream will be placed in
             * the final (end-of-stream) state.
             *
             * \pre the stream is not in the final state
             * \param[in] value to push or nullopt to indicate end of stream
             */
            void push_back(const std::optional<value_type>& value = {});


            /** \brief Is the final flag set?
             *
             * Note that even in the final state, the internal buffer may
             * contain elements that will be returned by calls to the pull
             * method.
             *
             * \return true if end-of-stream was seen
             */
            bool is_final() const noexcept;


            /** \brief If final size is unknown, pull all elements to 
             * determine size.
             *
             * If not in the end-of-stream state and if the final size is 
             * not known, then repeatedly call the handler to get and buffer
             * all elements until the end-of-stream signal is pulled.
             *
             * Any exceptions thrown by the handler are propagated out.
             * Values that were pulled before the exception are buffered.
             * The stream is left in a valid state.
             *
             * \return final size
             */
            size_type pull_size();


            /** \brief Pull all remaining elements and assemble into a final
             * container.
             *
             * If not in the end-of-stream state, repeatedly call the handler
             * to get and buffer all elements until the end-of-stream signal
             * is pulled.
             * Then assemble all buffered elements into a container of the
             * final_type.
             *
             * This method only assembles the elements presently in the buffer
             * to contruct the final container.
             * Any elements that have already been pulled are not included.
             *
             * This method does not remove any elements from the buffer
             * so they may be pulled or pull_final() may be called again with
             * the same result.
             *
             * Any exceptions thrown by the handler are propagated out.
             * Values that were pulled before the exception are buffered.
             * The stream is left in a valid state.
             *
             * \return final container
             */
            final_type pull_final();


            /** \brief Swap with other stream.
             */
            inline void swap(basic_puller& other) {
                state.swap(other.state);
            }


            /** \brief Write out description of stream in pseudo-json.
             */
            void describe(std::ostream& out, 
                          const std::string& indent = std::string()) const;
            

        protected:
            struct handler_base {
                virtual ~handler_base() {}
                virtual std::optional<value_type> operator()() = 0;
                virtual void describe(std::ostream& out,
                                      const std::string& indent) const;
            };
            template <typename HANDLER>
            struct handler_obj : public handler_base {
                HANDLER handler;
                handler_obj(const HANDLER& handler) : handler(handler) {}
                std::optional<value_type> operator()() override {
                    return handler();
                }
            };
            void set_handler_obj(std::unique_ptr<handler_base> handler,
                                 bool final = false);


            struct const_t {};


            /** \brief Constructor.
             *
             * If final_size is specified, the total data streamed must equal
             * this value when complete.
             *
             * \param[in] final_size optional final size specification
             */
            basic_puller(const std::optional<size_type>& final_size = {});

            /** \brief Constructor.
             *
             * Initialize to constant value and mark final.  
             * The final size is set to the size of the value.
             *
             * \param[in] value constant value
             */
            basic_puller(const value_type& value, const_t);



        private:
            struct internal;
            std::shared_ptr<internal> state;
        };

        template <typename T, typename FINAL, class OPS>
        inline void swap(basic_puller<T,FINAL,OPS>& a, 
                         basic_puller<T,FINAL,OPS>& b) {
            a.swap(b);
        }


        struct string_puller_ops;
        struct binary_puller_ops;
        struct array_puller_ops;
        struct object_puller_ops;
    }


    /** \brief Stream of string fragments.
     */
    class string_puller
        : public detail::basic_puller<string,string,
                                      detail::string_puller_ops> {
        using base_type =
            detail::basic_puller<string,string,detail::string_puller_ops>;
    public:
        explicit string_puller(const std::optional<size_type>& final_size = {})
            : base_type(final_size) {
        }
        string_puller(const string& v)
            : base_type(v,const_t()) {
        }
    };

    /** \brief Stream of binary fragments.
     */
    class binary_puller
        : public detail::basic_puller<binary,binary,
                                      detail::binary_puller_ops> {
        using base_type =
            detail::basic_puller<binary,binary,detail::binary_puller_ops>;
    public:
        explicit binary_puller(const std::optional<size_type>& final_size = {})
            : base_type(final_size) {
        }
        binary_puller(const binary& v)
            : base_type(v,const_t()) {
        }
    };

    /** \brief Stream of value_puller streams.
     */
    class array_puller
        : public detail::basic_puller<value_puller,array,
                                      detail::array_puller_ops> {
        using base_type =
            detail::basic_puller<value_puller,array,detail::array_puller_ops>;

        struct const_array;
        
    public:
        explicit array_puller(const std::optional<size_type>& final_size = {})
            : base_type(final_size) {
        }

        array_puller(const array& v);

        array_puller(const array& v,
                     array::const_iterator begin,
                     array::const_iterator end);

        using base_type::push_back;
        void push_back(const value& v);
    };

    /** \brief Stream of key and value_puller streams pairs.
     */
    class object_puller
        : public detail::basic_puller<std::pair<string,value_puller>,object,
                                      detail::object_puller_ops> {
        using base_type =
            detail::basic_puller<std::pair<string,value_puller>,object,
                                 detail::object_puller_ops>;

        struct const_object;
        
    public:
        explicit object_puller(const std::optional<size_type>& final_size = {})
            : base_type(final_size) {
        }

        object_puller(const object& v);

        using base_type::push_back;
        void push_back(const string& key, const value_puller& value);
    };


    namespace detail {
        using value_puller_base =
            std::variant<null_type, boolean, integer, real,
                         string_puller, binary_puller,
                         array_puller, object_puller>;

        template <typename T>
        using is_json_pull_type = stdx::is_one_of<
            T,
            null_type, boolean, integer, real,
            string_puller, binary_puller, array_puller, object_puller>;
    }


    /** \brief Basic value or stream type.
     */
    class value_puller : public detail::value_puller_base {
        using val = detail::value_puller_base;

    public:
        using final_type = value;
        
        /** \name Constructor
         */
        //@{
        constexpr value_puller() : val(null) {}
        constexpr value_puller(const null_type&) : val(null) {}

        // numeric (including bool and enum, but not char types)
        template <typename T>
        value_puller(T v,
                     std::enable_if_t<
                     stdx::is_bool<T>::value ||
                     std::is_enum<T>::value ||
                     stdx::is_pure_integral<T>::value ||
                     std::is_floating_point<T>::value>* = nullptr) 
            : val(detail::numeric_cast(v)) {}

        value_puller(const char* v) : val(string_puller(string(v))) {}
        value_puller(const string& v) : val(string_puller(v)) {}
        value_puller(const string_puller& v) : val(v) {}

        explicit value_puller(const binary& v) : val(binary_puller(v)) {}
        value_puller(const binary_puller& v) : val(v) {}

        value_puller(const array& v)
            : val(array_puller(v)) {}
        value_puller(const array& v,
                     array::const_iterator begin,
                     array::const_iterator end)
            : val(array_puller(v,begin,end)) {}
        value_puller(const array_puller& v)
            : val(v) {}

        value_puller(const object& v)
            : val(object_puller(v)) {}
        value_puller(const object_puller& v)
            : val(v) {}

        value_puller(const value& v);

        value_puller(value_puller&&) = default;
        value_puller(const value_puller&) = default;
        //@}


        /** \name Assignment
         */
        //@{
        value_puller& operator=(value_puller&&) = default;
        value_puller& operator=(const value_puller&) = default;
        inline void swap(value_puller& other) {
            static_cast<val*>(this)->swap(other);
        }
        //@}


        /** \name Access
         */
        //@{

        bool is_final() const noexcept;

        // pulls until final
        final_type pull_final();

        //@}


        /** \name Output
         */
        //@{
        /** \brief Write out description of stream in pseudo-json.
         */
        void describe(std::ostream& out, const std::string& indent = {}) const;
        //@}
    };

    
    /** \name Access Methods
     */
    //@{
    /// test for null
    inline bool is_null(const value_puller& v) {
        return v.index() == 0;
    }
    /// test for specific json type
    template <typename T>
    inline typename std::enable_if<detail::is_json_pull_type<T>::value,bool>::type
    is_type(const value_puller& v) {
        return std::holds_alternative<T>(v);
    }

    /// pretty name for type
    const char* type_name(const value_puller& v);

    /// same as std::get but only works with value_puller object
    template <typename T>
    inline std::enable_if_t<detail::is_json_pull_type<T>::value,T&>
    get(value_puller& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        throw bad_get(type_name<T>(), type_name(v));
    }
    template <typename T>
    inline std::enable_if_t<detail::is_json_pull_type<T>::value,T&&>
    get(value_puller&& v) {
        if (auto p = std::get_if<T>(&v))
            return move(*p);
        throw bad_get(type_name<T>(), type_name(v));
    }
    template <typename T>
    inline std::enable_if_t<detail::is_json_pull_type<T>::value,const T&>
    get(const value_puller& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        throw bad_get(type_name<T>(), type_name(v));
    }

    /// get boolean
    inline boolean get_boolean(const value_puller& v) {
        return get<boolean>(v);
    }
    /// get integer
    inline integer get_integer(const value_puller& v) {
        return get<integer>(v);
    }
    /// get real (converting from integer if necessary)
    real make_real(const value_puller& v);

    /// test for final
    inline bool is_final(const value_puller& v) noexcept {
        return v.is_final();
    }
    /// pull final value
    inline value pull_final(value_puller& v) {
        return v.pull_final();
    }

    /** \brief Get string_puller.
     *
     * If convert_cast, binary data is cast as string data.
     * If convert_base64, binary data is base64 encoded.
     */
    string_puller pull_string(const value_puller& val, 
                              convert_type convert = convert_none);

    /** \brief Get binary_puller.
     *
     * If convert_cast, string data is cast as binary data.
     * If convert_base64, string data is base64 decoded.
     */
    binary_puller pull_binary(const value_puller& val, 
                              convert_type convert = convert_none);

    /** \brief Get array_puller.
     */
    array_puller pull_array(const value_puller& val);

    /** \brief Get object_puller.
     */
    object_puller pull_object(const value_puller& val);
    //@}


    /** \name Describe Stream
     */
    //@{
    /** \brief Write out description of stream.
     */
    std::ostream& operator<<(std::ostream& out, const value_puller& stream);
    //@}

}


