#pragma once

#include <cstdint>
#include <utility>

template <std::size_t Offset, typename Seq>
struct offset_index_sequence_impl;

template <std::size_t Offset, std::size_t... Is>
struct offset_index_sequence_impl<Offset, std::index_sequence<Is...>> {
    using type = std::index_sequence<(Is + Offset)...>;
};

template <std::size_t Offset, std::size_t N>
using make_index_sequence_from = typename offset_index_sequence_impl<Offset, std::make_index_sequence<N>>::type;

template <std::size_t Start, std::size_t End>
using make_index_range = make_index_sequence_from<Start, End - Start>;