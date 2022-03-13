#pragma once

#include <memory>
#include <initializer_list>
#include <limits>
#include <assert.h>
#include <iterator>
#include <iostream>

// #include <vector>
// namespace bio{
//     // template <typename,typename> class Block;
//     template <typename T>
//     class Block : public std::vector<T>{};
//     // typedef typename std::vector<T> Block<T>;
// }

namespace bio{
    template <typename,typename> class Block;
    template <class T, class A = std::allocator<T> > void swap(Block<T,A>& lhs, Block<T,A>& rhs) noexcept;

    template <typename T, typename A = std::allocator<T>>
    class Block {
    public:
        typedef A allocator_type;
        typedef typename A::value_type value_type; 
        typedef typename A::value_type& reference;
        typedef const typename A::value_type& const_reference;
        typedef typename A::difference_type difference_type;
        typedef typename A::size_type size_type;

        class iterator;
        class const_iterator;
        // typedef typename A::value_type* iterator;
        // typedef const typename A::value_type* const_iterator;
        // typedef std::reverse_iterator<iterator> reverse_iterator;
        // typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    private:
        allocator_type alloc_;
        size_type size_ = 0;
        value_type *begin_ = nullptr;
        size_type capacity_ = 0;
        void _destroy() {
            value_type* q = begin_ + size_;
            while(q!=begin_)
                std::allocator_traits<allocator_type>::destroy(alloc_,--q);
            size_ = 0;
            // std::cout << "destroy!" << std::endl;
        }
        void _deallocate() {
            std::allocator_traits<allocator_type>::deallocate(alloc_,begin_,capacity_);
            capacity_ = 0;
            // std::cout << "deallocate!" << std::endl;
        }
    public:
        //iterator implementation
        class iterator {
        public:
            typedef typename A::difference_type difference_type;
            typedef typename A::value_type value_type;
            typedef typename A::value_type& reference;
            typedef typename A::pointer pointer;
            typedef std::random_access_iterator_tag iterator_category; //or another tag

            iterator();
            iterator(pointer p) : _current(p) {}
            iterator(const iterator& iter) : _current(iter._current) {}
            // ~iterator();

            iterator& operator=(const iterator& iter) {
                this->_current = iter._current;
                return *this;
            }
            bool operator==(const iterator& iter) const {
                return iter._current==this->_current;
            }
            bool operator!=(const iterator& iter) const {
                return iter._current!=this->_current;
            }
            bool operator<(const iterator& iter) const { //optional
                return this->_current < &(*iter);
            }
            bool operator>(const iterator& iter) const { //optional
                return this->_current > &(*iter);
            }
            bool operator<=(const iterator& iter) const { //optional
                return this->_current <= &(*iter);
            }
            bool operator>=(const iterator& iter) const { //optional
                return this->_current >= &(*iter);
            }

            iterator& operator++() {
                this->_current++;
                return *this;
            }
            iterator operator++(int) { //optional
                iterator iter(*this);
                operator++();
                return iter;
            }
            iterator& operator--() { //optional
                this->_current--;
                return *this;
            }
            iterator operator--(int) { //optional
                iterator iter(*this);
                operator--();
                return iter;
            }
            iterator& operator+=(size_type n) { //optional
                this->_current+=n;
                return *this;
            }
            iterator operator+(size_type n) const { //optional
                return iterator(this->_current+n);
            }
            // friend iterator operator+(size_type, const iterator&); //optional
            iterator& operator-=(size_type n) { //optional 
                this->_current-=n;
                return *this;
            }           
            iterator operator-(size_type n) const { //optional
                return iterator(this->_current-n);
            }
            difference_type operator-(iterator iter) const { //optional
                return this->_current - iter._current;
            }
            difference_type operator-(const_iterator iter) const { //optional
                return this->_current - &(*iter);
            }

            reference operator*() const {
                return *_current;
            }
            pointer operator->() const {
                return _current;
            }
            // reference operator[](size_type) const; //optional
        private:
            pointer _current;
        };
        class const_iterator {
        public:
            typedef typename A::difference_type difference_type;
            typedef typename A::value_type value_type;
            typedef const typename A::value_type& reference;
            typedef typename A::const_pointer pointer;
            typedef std::random_access_iterator_tag iterator_category; //or another tag

            const_iterator ();
            const_iterator(pointer p) : _current(p) {}
            const_iterator (const const_iterator& iter) : _current(iter._current) {}
            const_iterator (const iterator& iter) : _current(&(*iter)) {};
            // ~const_iterator();

            const_iterator& operator=(const const_iterator& iter) {
                this->_current = iter._current;
                return *this;
            }
            bool operator==(const const_iterator& iter) const {
                return iter._current==this->_current;
            }
            bool operator!=(const const_iterator& iter) const {
                return iter._current!=this->_current;
            }
            bool operator<(const const_iterator& iter) const { //optional
                return this->_current < &(*iter);
            }
            bool operator>(const const_iterator& iter) const { //optional
                return this->_current > &(*iter);
            }
            bool operator<=(const const_iterator& iter) const { //optional
                return this->_current <= &(*iter);
            }
            bool operator>=(const const_iterator& iter) const { //optional
                return this->_current >= &(*iter);
            }

            const_iterator& operator++() {
                this->_current++;
                return *this;
            }
            const_iterator operator++(int) {//optional
                const_iterator const_iter(*this);
                operator++();
                return const_iter;
            }
            const_iterator& operator--() { //optional
                this->_current--;
                return *this;
            }
            const_iterator operator--(int) { //optional
                const_iterator const_iter(*this);
                operator--();
                return const_iter;
            }
            const_iterator& operator+=(size_type n) { //optional
                this->_current+=n;
                return *this;
            }
            const_iterator operator+(size_type n) const { //optional
                return const_iterator(this->_current+n);
            }
            // friend const_iterator operator+(size_type, const const_iterator&); //optional
            const_iterator& operator-=(size_type n) { //optional   
                this->_current-=n;
                return *this;
            }         
            const_iterator operator-(size_type n) const { //optional
                return const_iterator(this->_current-n);
            }
            difference_type operator-(const_iterator iter) const { //optional
                return this->_current - iter._current;
            }

            reference operator*() const {
                return *_current;
            }
            pointer operator->() const {
                return _current;
            }
            // reference operator[](size_type) const; //optional
        private:
            pointer _current;
        };

        typedef std::reverse_iterator<iterator> reverse_iterator; //optional
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator; //optional

        //default
        Block() : begin_(nullptr), size_(0), capacity_(0) {
            // std::cout << "default contruct" << std::endl;
        }
        // explicit Block(allocator_type& alloc) {}
        //fill
        explicit Block(size_type n, allocator_type &&alloc = allocator_type()) : size_(n), capacity_(n), begin_(std::allocator_traits<allocator_type>::allocate(alloc,n)) {
            // std::cout << "fill contruct! 1" << std::endl;
            value_type* q = begin_;
            while(q!=(begin_ + size_))
                std::allocator_traits<allocator_type>::construct(alloc,q++);
        }
        explicit Block(size_type n, const value_type& val, allocator_type&& alloc = allocator_type()) : size_(n), capacity_(n), begin_(std::allocator_traits<allocator_type>::allocate(alloc,n)) {
            // std::cout << "fill contruct! 2" << std::endl;
            value_type* q = begin_;
            while(q!=(begin_ + size_))
                std::allocator_traits<allocator_type>::construct(alloc,q++,val);
        }
        // //range
        template <class InputIterator>
        Block(InputIterator first, InputIterator last, allocator_type&& alloc = allocator_type()) : size_(last-first), capacity_(last-first), begin_(std::allocator_traits<allocator_type>::allocate(alloc,last-first)) {
            // std::cout << "range constructor!" << std::endl;
            std::uninitialized_copy(&(*first),&(*last),this->begin_);
        }
        // //copy
        Block(const Block& x) {
            // std::cout << "copy constructor! 1" << std::endl;
            this->size_ = x.size_;
            this->capacity_ = this->size_;
            this->begin_ = std::allocator_traits<allocator_type>::allocate(alloc_,this->capacity_);
            std::uninitialized_copy(x.begin_,x.begin_+x.size_,this->begin_);
        }
        Block(const Block& x, allocator_type&& alloc) {
            // std::cout << "copy constructor! 2" << std::endl;
            this->size_ = x.size_;
            this->capacity_ = this->size_;
            this->begin_ = std::allocator_traits<allocator_type>::allocate(alloc,this->capacity_);
            std::uninitialized_copy(x.begin_,x.begin_+x.size_,this->begin_);
        }
        // //move
        Block(Block&& x) : begin_(nullptr), size_(0), capacity_(0) {
            // std::cout << "move constructor! 1" << std::endl;
            this->begin_ = x.begin_;
            this->size_ = x.size_;
            this->capacity_ = x.capacity_;

            x.begin_ = nullptr;
            x.size_ = 0;
            x.capacity_ = 0;
        }
        Block(Block&& x, allocator_type&& alloc) : begin_(nullptr), size_(0), capacity_(0) {
            // std::cout << "move constructor! 2" << std::endl;
            this->begin_ = x.begin_;
            this->size_ = x.size_;
            this->capacity_ = x.capacity_;

            x.begin_ = nullptr;
            x.size_ = 0;
            x.capacity_ = 0;
        }

        //initializer
        Block(std::initializer_list<value_type> il, allocator_type&& alloc = allocator_type()) {
            // std::cout << "initializer constructor!" << std::endl;
            this->size_ = il.size();
            this->capacity_ = this->size_;
            this->begin_ = std::allocator_traits<allocator_type>::allocate(alloc,this->capacity_);
            std::uninitialized_copy(il.begin(),il.end(),this->begin());
        }
        //desconstructor
        ~Block() {
            _destroy();
            _deallocate();
        }

        Block& operator=(const Block& x) {
            // std::cout << "copy assignment!" << std::endl;
            if(this != &x) {
                this->_destroy();
                this->_deallocate();
                size_ = x.size_;
                capacity_ = x.size_;
                this->begin_ = std::allocator_traits<allocator_type>::allocate(alloc_,capacity_);
                std::uninitialized_copy(x.begin_,x.begin_+x.size_,this->begin_);
            }
            return *this;
        }
        Block& operator=(Block&& x) {
            // std::cout << "move assignment!" << std::endl;
            if(this != &x) {
                this->_destroy();
                this->_deallocate();

                begin_ = x.begin_;
                size_ = x.size_;
                capacity_ = x.capacity_;

                x.begin_ = nullptr;
                x.size_ = 0;
                x.capacity_ = 0;
            }
            return *this;
        }
        Block& operator=(std::initializer_list<value_type> il) {
            // std::cout << "initializer_list assignment!" << std::endl;
            this->_destroy();
            this->_deallocate();
            size_ = il.size();
            capacity_ = size_;
            this->begin_ = std::allocator_traits<allocator_type>::allocate(alloc_,capacity_);
            std::uninitialized_copy(il.begin(),il.end(),this->begin());
            return *this;
        }
        

        iterator begin() noexcept {
            return iterator(begin_);
        }
        const_iterator begin() const noexcept {
            return const_iterator(begin_);
        }
        const_iterator cbegin() const noexcept {
            return const_iterator(begin_);
        }
        iterator end() noexcept {
            return iterator(begin_ + size_);
        }
        const_iterator end() const noexcept {
            return const_iterator(begin_ + size_);
        }
        const_iterator cend() const noexcept {
            return const_iterator(begin_ + size_);
        }
        reverse_iterator rbegin() noexcept { //optional
            return reverse_iterator(begin_ + size_);
        }
        const_reverse_iterator rbegin() const noexcept { //optional
            return const_reverse_iterator(begin_ + size_);
        }
        const_reverse_iterator crbegin() const noexcept { //optional
            return const_reverse_iterator(begin_ + size_);
        }
        reverse_iterator rend() noexcept { //optional
            return reverse_iterator(begin_);
        }
        const_reverse_iterator rend() const noexcept { //optional
            return const_reverse_iterator(begin_);
        }
        const_reverse_iterator crend() const noexcept { //optional
            return const_reverse_iterator(begin_);
        }

        // reference front(); //optional
        // const_reference front() const; //optional
        // reference back(); //optional
        // const_reference back() const; //optional
        // template<class ...Args>
        // void emplace_front(Args&&...); //optional
        // template<class ...Args>
        // void emplace_back(Args&&...); //optional
        // void push_front(const T&); //optional
        // void push_front(T&&); //optional
        void push_back(const value_type& val) {//optional
            // std::cout << "const reference push_back" << std::endl;
            if(size_ >= capacity_) reserve(2*capacity_+1);
            // begin_[size_] = val;
            std::allocator_traits<allocator_type>::construct(alloc_,begin_+size_,val);
            ++size_;
        }
        void push_back(value_type&& val) { //optional
            // std::cout << "move push_back" << std::endl;
            if(size_ >= capacity_) reserve(2*capacity_+1);
            // begin_[size_] = val;
            std::allocator_traits<allocator_type>::construct(alloc_,begin_+size_,val);
            ++size_;
        }
        // void pop_front(); //optional
        // void pop_back(); //optional
        reference operator[](size_type n) { //optional
            return *(begin_+n);
        }
        const_reference operator[](size_type n) const { //optional
            return *(begin_+n);
        }
        // reference at(size_type); //optional
        // const_reference at(size_type) const; //optional

        // template<class ...Args>
        // iterator emplace(const_iterator, Args&&...); //optional
        iterator insert(const_iterator position, const value_type& val) { //optional
            // std::cout << "single element insert" << std::endl;
            assert(&(*position) <= (this->begin_+this->size_) && &(*position) >= this->begin_);
            const auto pos = position-const_iterator(begin_);
            push_back(val);
            // value_type&& tmp_T = std::move(*(this->begin_+this->size_-1));
            const_iterator iter = const_iterator(this->begin_+this->size_-1);
            while(iter!=(const_iterator(begin_)+pos)) {
                std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
                std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(iter-1));
                --iter;
            }
            std::allocator_traits<allocator_type>::destroy(alloc_,&(*(const_iterator(begin_)+pos)));
            std::allocator_traits<allocator_type>::construct(alloc_,&(*(const_iterator(begin_)+pos)),val);
            return iterator(begin_)+pos;
        }
        iterator insert(const_iterator position, value_type&& val) { //optional
            // std::cout << "move insert" << std::endl;
            assert(&(*position) <= (this->begin_+this->size_) && &(*position) >= this->begin_);
            const auto pos = position-const_iterator(begin_);
            push_back(val);
            // if(&(*position)==) return iterator(begin_+size_-1);
            // value_type&& tmp_T = std::move(*(this->begin_+this->size_-1));
            const_iterator iter = const_iterator(this->begin_+this->size_-1);
            while(iter!=(const_iterator(begin_)+pos)) {
                std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
                std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(iter-1));
                --iter;
            }
            std::allocator_traits<allocator_type>::destroy(alloc_,&(*(const_iterator(begin_)+pos)));
            std::allocator_traits<allocator_type>::construct(alloc_,&(*(const_iterator(begin_)+pos)),val);
            return iterator(begin_)+pos;
        }
        iterator insert(const_iterator position, size_type n, value_type& val) { //optional
            // std::cout << "fill insert" << std::endl;
            assert(n>=0);
            const auto pos = position-const_iterator(begin_);
            for(auto i = 0; i != n; ++i) push_back(val);
            const_iterator iter = const_iterator(begin_+size_)-1;
            while(iter!=(const_iterator(begin_)+pos-1)) {
                std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
                if(iter-(const_iterator(begin_)+pos) >= n) { 
                    std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(iter-n));
                }else{
                    std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),val);
                }
                --iter;
            }
            return iterator(begin_)+pos;
        }
        template<class InputIterator>
        iterator insert(const_iterator position, InputIterator first, InputIterator last) { //optional
            // std::cout << "range insert" << std::endl;
            typename std::iterator_traits<InputIterator>::difference_type range = std::distance(first,last);
            assert(range>=0);
            InputIterator moving_last = last-1;
            const auto pos = position-const_iterator(begin_);
            for(auto i = first; i != last; ++i) push_back(*i);
            const_iterator iter = const_iterator(begin_+size_)-1;
            while(iter!=(const_iterator(begin_)+pos-1)) {
                std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
                if(iter-(const_iterator(begin_)+pos) >= range) { 
                    std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(iter-range));
                }else{
                    std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(moving_last));
                    --moving_last;
                }
                --iter;
            }
            return iterator(begin_)+pos;
        }
        iterator insert(const_iterator position, std::initializer_list<value_type> il) { //optional
            // std::cout << "initializer list insert" << std::endl;
            return insert(position,il.begin(),il.end());
        }


        iterator erase(const_iterator position) { //optional
            // std::cout << "single erase" << std::endl;
            const_iterator iter = position;
            while(iter!=(this->begin_+this->size_-1)) {
                std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
                std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(iter+1));
                ++iter;
            }
            std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
            --size_;
            return iterator(begin_)+(position-const_iterator(begin_));
        }
        iterator erase(const_iterator first, const_iterator last) { //optional
            //last will not be erased
            // std::cout << "range erase" << std::endl;
            assert(first < last);
            auto range = last-first;
            const_iterator iter = first;
            const_iterator iter_end = const_iterator(begin_+size_);
            while(iter!=iter_end) {
                std::allocator_traits<allocator_type>::destroy(alloc_,&(*iter));
                if(iter_end-iter > range) { 
                    std::allocator_traits<allocator_type>::construct(alloc_,&(*iter),*(iter+range));
                }else{
                    --size_;
                }
                ++iter;
            }
            return iterator(begin_+(first-const_iterator(begin_)));
        }
        void clear() noexcept {
            _destroy();
        }
        // template<class iter>
        // void assign(iter, iter); //optional
        // void assign(std::initializer_list<T>); //optional
        // void assign(size_type, const T&); //optional

        void reserve(size_type n) {
            if(n<=capacity_) return;
            value_type* const new_begin = std::allocator_traits<allocator_type>::allocate(alloc_,n); // allocate a new array on the free store
            std::uninitialized_copy(begin_,begin_+size_,new_begin);
            size_type old_size = size_;
            _destroy();
            _deallocate();
            size_ = old_size;
            capacity_ = n;
            begin_ = new_begin;
        }

        size_type size() const noexcept {
            return this->size_;
        }
        size_type capacity() const noexcept {
            return this->capacity_;
        }
        size_type max_size() const noexcept {
            return std::numeric_limits<size_type>::max() / sizeof(value_type);
        }
        bool empty() const noexcept {
            return this->size_ == 0;
        }

        allocator_type get_allocator() const noexcept { //optional
            return alloc_;
        }

        value_type* data() noexcept {
            return begin_;
        }
        const value_type* data() const noexcept {
            return begin_;
        }

        friend void swap<T,A>(Block<T,A>&, Block<T,A>&) noexcept;

        void swap(Block& x) noexcept {
            bio::swap(*this,x);
        }

        
    };
    template <class T, class A = std::allocator<T> >
    void swap(Block<T,A>& lhs, Block<T,A>& rhs) noexcept { //optional
            T *tmp_begin = lhs.begin_;
            typename A::size_type tmp_size = lhs.size_;
            typename A::size_type tmp_capacity = lhs.capacity_;

            lhs.begin_ = rhs.begin_;
            lhs.size_ = rhs.size_;
            lhs.capacity_ = rhs.capacity_;

            rhs.begin_ = tmp_begin;
            rhs.size_ = tmp_size;
            rhs.capacity_ = tmp_capacity;
    }

    template <class T, class A = std::allocator<T> >
    bool operator==(const Block<T,A>& lhs, const Block<T,A>& rhs) {
        if(lhs.size() != rhs.size()) return false;
        for(auto i = 0; i != lhs.size(); ++i)
        {
            if(lhs[i]!=rhs[i]) return false;
        }
        return true;
    }
    template <class T, class A = std::allocator<T> >
    bool operator!=(const Block<T,A>& lhs, const Block<T,A>& rhs) {
        return !(lhs==rhs);
    }
    // template <class T, class A = std::allocator<T> >
    // bool operator<(const Block<T,A>& lhs, const Block<T,A>& rhs); //optional
    // template <class T, class A = std::allocator<T> >
    // bool operator>(const Block<T,A>& lhs, const Block<T,A>& rhs); //optional
    // template <class T, class A = std::allocator<T> >
    // bool operator<=(const Block<T,A>& lhs, const Block<T,A>& rhs); //optional
    // template <class T, class A = std::allocator<T> >
    // bool operator>=(const Block<T,A>& lhs, const Block<T,A>& rhs); //optional
    
}//namespace bio