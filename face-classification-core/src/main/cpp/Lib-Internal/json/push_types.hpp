#pragma once

#include <memory>

#include <optional>
#include <stdext/forward_iterator.hpp>

#include "types.hpp"


namespace json {

    namespace detail {


        /** \brief Pusher type stream for container.
         *
         * The internal state of the stream includes the following: <ul>
         *   <li>an buffer of values</li>
         *   <li>an optional final size</li>
         *   <li>a final flag</li>
         *   <li>a handler to which to push values</li>
         *   <li>a handler to which to send a final container</li>
         *   <li>one or more parent handlers to call at end-of-stream</li>
         * </ul>
         *
         * Values will be pushed through immediately if a push handler is
         * set; however, if no push handler is set or if one of the values
         * pushed through is a pusher that is not in the final state, then
         * values will be stored in the buffer.
         * When the end-of-stream signal is pushed, the final flag is set and,
         * if the stream is not waiting on an incomplete child stream, the
         * end-of-stream signal is pushed through by calling the push handler,
         * the final handler and any parent handlers.
         *
         * The final_size may be specified in advance or will be determined
         * once the end-of-stream signal is seen.
         * If the final size is set and an attempt is made to push more data
         * than this size, an exception is thrown.  
         * If the stream is shorter than the final size, an error is logged.
         * The end-of-stream signal is required even if the final size is
         * known in advance.
         */
        template <typename T, typename FINAL, class OPS>
        class basic_pusher {

        public:
            /** \brief Streamed element type.
             */
            using value_type = T;

            /** \brief Type for final container of elements.
             */
            using final_type = FINAL;

            /** \brief Iterator for range of elements.
             */
            using iterator = typename stdx::forward_iterator<T>;

            /** \brief Result of push operator.
             */
            using result_type = void;

            /** \brief Integer size type.
             */
            using size_type = std::size_t;

            
            /** \brief Set final size.
             *
             * This method may not be called if a final size has already known
             * (unless the set value is the same as the known size)
             * or if more than the specified size has already passed through.
             *
             * If an attempt is made to push more data than indicated by this
             * final size, the push method will throw an exception. 
             * If less data than this final size has been pushed when the
             * end-of-stream signal is seen, an error is logged.
             *
             * The metric used to tally the final size is type dependent.
             * For array and object types it is the number of elements passed
             * while for string and binary types it is the number of characters
             * or bytes passed.
             *
             * \param[in] final_size final size specification
             */
            void set_final_size(size_type final_size);

            /** \brief Final stream size (if known).
             *
             * The metric used to tally the final size is type dependent.
             * For array and object types it is the number of elements passed
             * while for string and binary types it is the number of characters
             * or bytes passed.
             *
             * \return final size or nullopt if unknown
             */
            const std::optional<size_type>& final_size() const;

            /** \brief Is the stream in the final state?
             *
             * This method returns true if the the end-of-stream signal has
             * been seen and the stream is not currently waiting on any
             * contained stream to finish.
             *
             * Note that even in the final state, the internal buffer may
             * contain elements that will be passed to a push handler or
             * final handler when set.
             *
             * \return true if in final state
             */
            bool is_final() const noexcept;

            /** \brief Get final container with buffer content.
             *
             * If the stream is not yet in the final state an exception is
             * thrown.
             * Any values in the internal buffer are assembled into a container
             * of the final type and returned.
             * The value is moved from the pusher object so this method
             * should only be called once before destructing the pusher.
             *
             * \return final container
             */
            final_type take_final();

            /** \brief Get final container with buffer content.
             *
             * If the stream is not yet in the final state an exception is
             * thrown.
             * Any values in the internal buffer are assembled into a container
             * of the final type and returned.
             * Values are not removed from the buffer.
             *
             * \return final container
             */
            const final_type& final_value() const;


            /** \brief Set handler to push data to.
             *
             * The handler must take a single argument that can be constructed
             * from nullopt or value_type
             * (i.e. std::optional<value_type>).
             * A value of nullopt is pushed to indicate end-of-stream.
             *
             * If any values are present in the internal buffer, they will be
             * immediately pushed by calling the handler from within this
             * method.
             * If the stream is in the final state, the end-of-stream signal
             * is also pushed.  
             * The handler is cleared after the end-of-stream signal is pushed.
             *
             * If the handler throws an exception, the exception will be
             * propagated out and the handler will not be set.
             * Any values successfully processed by the handler are removed
             * from the internal buffer.
             * The value being passed to the handler when the exception was
             * thrown will remain in the buffer.
             *
             * The handler may be copied when set, but will not be copied
             * again once it starts being called.
             *
             * \pre no push handler is currently set
             * \param[in] handler push handler
             */
            template <typename HANDLER>
            inline void set_value_handler(const HANDLER& handler) {
                set_value_handler_obj(
                    std::make_unique<value_handler_obj<HANDLER> >(handler));
            }

            /** \brief Set handler to push data to.
             *
             * The handler must take two arguments of iterator type.
             * If the two arguments are non-equal, they represent a range
             * of elements.
             * Otherwise, an empty range indicates end-of-stream.
             *
             * If any values are present in the internal buffer, they will be
             * immediately pushed by calling the handler from within this
             * method.
             * If the stream is in the final state, the end-of-stream signal
             * is also pushed.  
             * The handler is cleared after the end-of-stream signal is pushed.
             *
             * If the handler throws an exception, the exception will be
             * propagated out and the handler will not be set.
             * If the exception was thrown while processing values from the
             * buffer, the values currently in the buffer will remain.
             * If the content of the buffer are successfully processed by
             * the handler (the exception happened when the end-of-stream
             * signal was pushed), then the values are removed from the
             * buffer.
             *
             * The handler may be copied when set, but will not be copied
             * again once it starts being called.
             *
             * \pre no push handler is currently set
             * \param[in] handler push handler
             */
            template <typename HANDLER>
            inline void set_range_handler(const HANDLER& handler) {
                set_range_handler_obj(
                    std::make_unique<range_handler_obj<HANDLER> >(handler));
            }

            /** \brief Set handler to send final data to.
             *
             * The handler must take a single argument that can be constructed
             * from an object of type final_type.
             * If the stream is in the final state, the handler will be called
             * immediately and passed a container containing any data 
             * currently held in the internal buffer.
             *
             * An exception thrown by the handler will be propagated out and
             * the internal state of the stream will not have changed.
             *
             * Regardless of the outcome of the call to the handler, the
             * content of the internal buffer will not have changed and
             * the handler will be cleared.
             *
             * \pre no final handler is currently set
             * \param[in] handler final handler
             */
            template <typename HANDLER>
            inline void set_final_handler(
                const HANDLER& handler,
                object::key_compare comp = object::key_compare()) {
                set_final_handler_obj(
                    std::make_unique<final_handler_obj<HANDLER> >(handler),
                    comp);
            }

            /** \brief Set handler to call when stream is complete.
             *
             * Parent handlers are called after the final handler (if set)
             * when a stream completes.  
             * If multiple parent handler have been set, they are called
             * in the reverse of the order they were set.
             *
             * \pre the stream is not in the final state
             * \param[in] handler final handler
             */
            template <typename HANDLER>
            inline void set_parent_handler(const HANDLER& handler) {
                set_parent_handler_obj(
                    std::make_unique<parent_handler_obj<HANDLER> >(handler));
            }


            /** \brief Push end-of-stream signal.
             *
             * If the stream is waiting on a contained stream that is not yet
             * complete, this method essentially queues the end-of-stream
             * signal.
             * Otherwise, it will signal end-of-stream by doing the following:
             * <ul>
             *   <li>set the final flag</li>
             *   <li>set final size if not already set (log error if stream
             *       is too short)</li>
             *   <li>call push handler with end-of-stream signal (if set)</li>
             *   <li>call final handler (if set)</li>
             *   <li>call any parent handlers</li>
             * </ul>
             *
             * All of the handlers will called and cleared even if some of
             * them throw exceptions.  
             * If exceptions are thrown, the exceptions are logged and the
             * last exception to be thrown is propagated out.
             *
             * \pre final flag is not set
             */
            void operator()();

            /** \brief Push value or end-of-stream signal.
             *
             * If nullopt is pushed, signal end-of-stream as described in
             * operator()().
             *
             * If no push handler is set or if the stream is waiting for an
             * earlier contained stream to complete, the value is stored
             * in the internal buffer.
             * Otherwise, the push handler is called to handle the value.
             *
             * If the push handler throws an exception, the value is stored
             * in the internal buffer, the push handler is cleared, and the
             * exception is propagated out.
             * Note that the value counts towards the final size even if an
             * exception is thrown.
             *
             * \pre final flag is not set
             * \param[in] val value to push
             */
            void operator()(std::optional<T> val);

            /** \brief Push range or end-of-stream signal.
             *
             * If begin == end, signal end-of-stream as described in
             * operator()().
             *
             * If no push handler is set or if the stream is waiting for an
             * earlier contained stream to complete, the values are stored
             * in the internal buffer.
             * Otherwise, the push handler is called to handle the values.
             *
             * If the push handler throws an exception, the values are stored
             * in the internal buffer, the push handler is cleared, and the
             * exception is propagated out.
             * Note that the values counts towards the final size even if an
             * exception is thrown.
             *
             * \pre final flag is not set
             * \param[in] begin start of range
             * \param[in] end end of range
             */
            void operator()(iterator begin, iterator end);

           
            /** \brief Swap with other stream.
             */
            inline void swap(basic_pusher& other) {
                state.swap(other.state);
            }


        protected:
            struct value_handler_base {
                virtual ~value_handler_base() {}
                virtual void operator()() = 0;
                virtual void operator()(const value_type&) = 0;
            };
            void set_value_handler_obj(
                std::unique_ptr<value_handler_base> handler);

            template <typename HANDLER>
            struct value_handler_obj : public value_handler_base {
                HANDLER handler;
                value_handler_obj(const HANDLER& handler) : handler(handler) {}
                void operator()() override {
                    handler(std::nullopt);
                }
                void operator()(const value_type& value) override {
                    handler(value);
                }
            };

            struct range_handler_base {
                virtual ~range_handler_base() {}
                virtual void operator()(iterator begin, iterator end) = 0;
            };
            void set_range_handler_obj(
                std::unique_ptr<range_handler_base> handler);

            template <typename HANDLER>
            struct range_handler_obj : public range_handler_base {
                HANDLER handler;
                range_handler_obj(const HANDLER& handler) : handler(handler) {}
                void operator()(iterator begin, iterator end) override {
                    handler(begin,end);
                }
            };
            
            struct final_handler_base {
                virtual ~final_handler_base() {}
                virtual void operator()(const final_type&) = 0;
            };
            void set_final_handler_obj(
                std::unique_ptr<final_handler_base> handler,
                object::key_compare comp);

            template <typename HANDLER>
            struct final_handler_obj : public final_handler_base {
                HANDLER handler;
                final_handler_obj(const HANDLER& handler) : handler(handler) {}
                void operator()(const final_type& val) override {
                    handler(val);
                }
            };

            struct parent_handler_base {
                virtual ~parent_handler_base() {}
                virtual void operator()() = 0;
            };
            void set_parent_handler_obj(
                std::unique_ptr<parent_handler_base> handler);
            
            template <typename HANDLER>
            struct parent_handler_obj : public parent_handler_base {
                HANDLER handler;
                parent_handler_obj(const HANDLER& handler) : handler(handler) {}
                void operator()() override {
                    handler();
                }
            };


            basic_pusher(
                const std::optional<size_type>& final_size);
            
            basic_pusher(const final_type& val);

            struct internal;
            std::shared_ptr<internal> state;
        };

        struct string_pusher_ops;
        struct binary_pusher_ops;
        struct array_pusher_ops;
        struct object_pusher_ops;


        using value_pusher_base =
            std::variant<null_type, boolean, integer, real,
                         string_pusher, binary_pusher,
                         array_pusher, object_pusher>;

        template <typename T>
        using is_json_push_type = stdx::is_one_of<
            T,
            null_type, boolean, integer, real,
            string_pusher, binary_pusher, array_pusher, object_pusher>;
    }

    /** \brief Stream of string fragments.
     */
    class string_pusher
        : public detail::basic_pusher<string,string,
                                      detail::string_pusher_ops> {
        using base_type =
            detail::basic_pusher<string,string,detail::string_pusher_ops>;
        
    public:
        explicit string_pusher(const std::optional<size_type>& size = {})
            : base_type(size) {}

        string_pusher(const string& val)
            : base_type(val) {}
    };


    /** \brief Stream of binary fragments.
     */
    class binary_pusher
        : public detail::basic_pusher<binary,binary,
                                      detail::binary_pusher_ops> {
        using base_type =
            detail::basic_pusher<binary,binary,detail::binary_pusher_ops>;
        
    public:
        explicit binary_pusher(const std::optional<size_type>& size = {})
            : base_type(size) {}

        binary_pusher(const binary& val)
            : base_type(val) {}
    };


    /** \brief Stream of value_pusher streams.
     */
    class array_pusher
        : public detail::basic_pusher<value_pusher,array,
                                      detail::array_pusher_ops> {
        using base_type =
            detail::basic_pusher<value_pusher,array,detail::array_pusher_ops>;
        
    public:
        explicit array_pusher(const std::optional<size_type>& size = {})
            : base_type(size) {}

        array_pusher(const array& val)
            : base_type(val) {}
    };


    /** \brief Stream of key and value_pusher streams pairs.
     */
    class object_pusher
        : public detail::basic_pusher<std::pair<string,value_pusher>,object,
                                      detail::object_pusher_ops> {
        using base_type =
            detail::basic_pusher<std::pair<string,value_pusher>,object,
                                 detail::object_pusher_ops>;

    public:
        explicit object_pusher(const std::optional<size_type>& size = {})
            : base_type(size) {}

        object_pusher(const object& val)
            : base_type(val) {}

        object take_final(object::key_compare comp = {});
        const object& final_value(object::key_compare comp = {}) const;
        
        void push_value(const string& key, const value& val);

    };


    /** \brief A streamable json value (a literal or one of the above streams).
     */
    class value_pusher : public detail::value_pusher_base {
        using val = detail::value_pusher_base;

    public:
        using final_type = value;

        /** \name Constructor
         */
        //@{
        constexpr value_pusher() : val(null) {}
        constexpr value_pusher(const null_type&) : val(null) {}

        // numeric (including bool and enum, but not char types)
        template <typename T>
        value_pusher(T v,
                     std::enable_if_t<
                     stdx::is_bool<T>::value ||
                     std::is_enum<T>::value ||
                     stdx::is_pure_integral<T>::value ||
                     std::is_floating_point<T>::value>* = nullptr) 
            : val(detail::numeric_cast(v)) {}

        value_pusher(const char* v) : val(string_pusher(string(v))) {}
        value_pusher(const string& v) : val(string_pusher(v)) {}
        value_pusher(const string_pusher& v) : val(v) {}
        
        explicit value_pusher(const binary& v) : val(binary_pusher(v)) {}
        value_pusher(const binary_pusher& v) : val(v) {}

        value_pusher(const array& v) 
            : val(array_pusher(v)) {}
        value_pusher(const array_pusher& v) 
            : val(v) {}
        
        value_pusher(const object& v) 
            : val(object_pusher(v)) {}
        value_pusher(const object_pusher& v) 
            : val(v) {}
        
        value_pusher(const value& v);

        value_pusher(value_pusher&&) = default;
        value_pusher(const value_pusher&) = default;
        //@}


        /** \name Assignment
         */
        //@{
        value_pusher& operator=(value_pusher&&) = default;
        value_pusher& operator=(const value_pusher&) = default;
        inline void swap(value_pusher& other) {
            static_cast<val*>(this)->swap(other);
        }
        //@}


        /** \name Access
         */
        //@{
        bool is_final() const noexcept;

        // throws if not final
        final_type take_final(object::key_compare comp = {});
        final_type final_value(object::key_compare comp = {}) const;
        //@}



        /** \brief Handler to call when end-of-stream is seen.
         *
         * If the stream is already complete, this handler will be called
         * immediately.  
         * Otherwise, this handler is called as a result of pushing the
         * end-of-stream signal.  
         *
         * An exception thrown by the handler is propagated out.
         *
         * This handler is cleared after it is called.
         *
         * Must #include <json/visit.hpp> to use this method.
         *
         * \param[in] handler final handler
         */
        template <typename HANDLER>
        void set_final_handler(HANDLER handler, object::key_compare comp = {});


        /** \brief Handler to call when end-of-stream is seen
         * (after final_handler).
         *
         * Must #include <json/visit.hpp> to use this method.
         *
         * \param[in] handler parent handler
         */
        template <typename HANDLER>
        void set_parent_handler(const HANDLER& handler);


    private:
        friend struct detail::array_pusher_ops;
        friend struct detail::object_pusher_ops;
    };


    /** \name Access Methods
     */
    //@{
    /// test for null
    inline bool is_null(const value_pusher& v) {
        return v.index() == 0;
    }
    /// test for specific json type
    template <typename T>
    inline std::enable_if_t<detail::is_json_push_type<T>::value,bool>
    is_type(const value_pusher& v) {
        return std::holds_alternative<T>(v);
    }

    /// pretty name for type
    const char* type_name(const value_pusher& v);

    /// same as std::get but only works with value_pusher object
    template <typename T>
    inline std::enable_if_t<detail::is_json_push_type<T>::value, T&>
    get(value_pusher& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        throw bad_get(type_name<T>(), type_name(v));
    }
    template <typename T>
    inline std::enable_if_t<detail::is_json_push_type<T>::value, T&&>
    get(value_pusher&& v) {
        if (auto p = std::get_if<T>(&v))
            return move(*p);
        throw bad_get(type_name<T>(), type_name(v));
    }
    template <typename T>
    inline std::enable_if_t<detail::is_json_push_type<T>::value, const T&>
    get(const value_pusher& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        throw bad_get(type_name<T>(), type_name(v));
    }

    /// test if final
    inline bool is_final(const value_pusher& v) noexcept {
        return v.is_final();
    }
    /// get final value
    inline value move_value(value_pusher& v, object::key_compare comp = {}) {
        return v.take_final(comp);
    }
    inline value copy_value(const value_pusher& v, object::key_compare comp = {}) {
        return v.final_value(comp);
    }

    /// get boolean
    inline boolean get_boolean(const value_pusher& v) {
        return get<boolean>(v);
    }
    /// get integer
    inline integer get_integer(const value_pusher& v) {
        return get<integer>(v);
    }
    /// get real (converting from integer if necessary)
    real make_real(const value_pusher& v);

    /** \brief Get string_pusher.
     *
     * If convert_cast, binary data is cast as string data.
     * If convert_base64, binary data is base64 encoded.
     */
    string_pusher get_string_pusher(const value_pusher& val,
                                    convert_type convert = convert_none);

    /** \brief Get binary_pusher.
     *
     * If convert_cast, string data is cast as binary data.
     * If convert_base64, string data is base64 decoded.
     */
    binary_pusher get_binary_pusher(const value_pusher& val,
                                    convert_type convert = convert_none);

    /** \brief Get array_pusher.
     */
    array_pusher get_array_pusher(const value_pusher& val);

    /** \brief Get object_pusher.
     */
    object_pusher get_object_pusher(const value_pusher& val);
    //@}

}


