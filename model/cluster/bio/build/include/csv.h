#pragma once

// #include <iterator>
#include <iostream>
// #include <fstream>
// #include <sstream>
#include <vector>
#include <string>

namespace util{
    class CSVRow{
        friend std::istream& operator>>(std::istream& str, CSVRow& data);
        friend std::ostream& operator<<(std::ostream& str, const CSVRow& data);
        public:
            char delimiter;
            char quotechar;
            CSVRow(char init_delimiter = ',', char init_quotechar = '\"');
            CSVRow(std::vector<std::string>& row, char init_delimiter = ',', char init_qutoechar = '\"' );
            // CSVRow( const CSVRow & );//copy constructor// if not define this, the functor ()overload won't work
            std::string const& operator[](std::size_t index) const;
            std::size_t size() const;
            CSVRow operator()(const std::vector<std::string>& row);
        private:
            std::vector<std::string> m_data;
            void readNextRow(std::istream& str);
            void writeNextRow(std::ostream& str);
    };

    std::istream& operator>>(std::istream& str, CSVRow& data);
    std::ostream& operator<<(std::ostream& str, const CSVRow& data);
    // std::ostream& operator<<(std::ostream& str, std::vector<std::string> &data);
}//namespace