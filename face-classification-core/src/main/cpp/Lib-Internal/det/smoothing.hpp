#pragma once

#include <utility>

namespace smoothing {

    /** \brief Double or Single Exponential Smoothing
     *
     * https://en.wikipedia.org/wiki/Exponential_smoothing
     *
     * The two parameters take values between 0 and 1.  
     * Lower values yield greater smoothing.
     * The alpha parameter controls smoothing of the value while 
     * the beta parameter controls smoothing of the rate.
     *
     * Use beta = 0 for single (simple) exponential smoothing.
     *
     * Use alpha = 1 to disable smoothing of the value.
     *
     * As an example starting point, try alpha 0.5 and beta 0.5.
     * Then for greater smoothing of the value while maintaining quick
     * response try lowering alpha to 0.25.
     *
     * To use with a custom value_type, the following operators are required:
     *   value_type& operator-=(value_type&, const value_type&)
     *   value_type& operator*=(value_type&, parameter_type)
     */
    template <typename VALUE_T = float, typename PARAM_T = float>
    class exponential {
    public:
        using parameter_type = PARAM_T;
        using value_type = VALUE_T;

    private:
        parameter_type m_alpha, m_am1, m_beta;
        value_type m_value, m_rate;
        bool m_valid;

    public:
        /** \brief Constructor.
         *
         * If beta is 0 (default) then single exponential smoothing.
         *
         * Value is invalid / undefined after construction.
         * Must call update() or set() before accessing value.
         */
        constexpr
        exponential(parameter_type alpha, parameter_type beta = PARAM_T{})
            : m_alpha(alpha), m_am1(alpha-1),
              m_beta(beta),
              m_value(),
              m_rate(),
              m_valid(false) {
        }

        /** \brief Access to parameters.
         */
        constexpr inline parameter_type alpha() const {
            return m_alpha;
        }
        constexpr inline parameter_type beta() const {
            return m_beta;
        }

        /** \brief Access to value and rate.
         *
         * Must update() or set() before accessing value.
         */
        constexpr inline operator const value_type&() const {
            return m_value;
        }
        constexpr inline const value_type& value() const {
            return m_value;
        }
        constexpr inline const value_type& rate() const {
            return m_rate;
        }

        /** \brief Update value.
         *
         * If this method is called when the value is undefined,
         * the value will be set (with rate 0).
         */
        const value_type& update(value_type sample) {
            if (!m_valid) {
                m_value = std::move(sample);
                m_valid = true;
            }
            else {
                // new_value = (1-alpha)*(old_value+old_rate) + alpha*sample
                // new_value = (1-alpha)*(old_value+old_rate + alpha/(1-alpha)*sample)
                if (m_am1 < 0) {
                    sample *= m_alpha / m_am1;
                    sample -= m_value;
                    sample -= m_rate;
                    sample *= m_am1;
                }
                auto& new_value = sample;

                // new_rate = (1-beta)*old_rate - beta*(old_value-new_value)
                if (m_beta > 0) {
                    m_value -= new_value;
                    m_value *= -m_beta;
                    m_rate *= m_beta-1;
                    m_value -= m_rate;
                    m_rate = std::move(m_value);
                }

                m_value = std::move(new_value);
            }
            return m_value;
        }
        
        /** \brief Set value and optionally rate.
         *
         * Calling this method is optional as the value may also be set
         * from undefined by calling update().
         */
        inline void set(value_type sample, value_type rate = VALUE_T{}) {
            m_rate = std::move(rate);
            m_value = std::move(sample);
            m_valid = true;
        }
        
        /** \brief Reset to undefined value.
         */
        inline void reset() {
            m_valid = false;
        }
    };
}
