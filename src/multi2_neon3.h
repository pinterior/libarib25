#pragma once

#include <arm_neon.h>
#include "portable.h"

namespace multi2 {

namespace arm {

template<size_t S>
struct neon3 {
  uint32x4_t a;
  uint32x4_t b;
  uint32x4_t c;

  inline neon3() { }

  inline neon3(const uint32x4_t &x, const uint32x4_t &y, const uint32x4_t &z) {
    a = x;
    b = y;
    c = z;
  }

  static inline void load_block(block<neon3> &b, const uint8_t *p) {
    size_t d = 8 - (12 - S) * 2;
    const uint32_t *q = reinterpret_cast<const uint32_t *>(p);
    uint32x4x2_t a0 = vld2q_u32(q);
    uint32x4x2_t a1 = vld2q_u32(q + d);
    uint32x4x2_t a2 = vld2q_u32(q + d + 8);

    uint32x4_t b0 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a0.val[0])));
    uint32x4_t b1 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a0.val[1])));
    uint32x4_t b2 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a1.val[0])));
    uint32x4_t b3 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a1.val[1])));
    uint32x4_t b4 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a2.val[0])));
    uint32x4_t b5 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a2.val[1])));

    b.left  = neon3(b0, b2, b4);
    b.right = neon3(b1, b3, b5);
  }

  static inline void store_block(uint8_t *p, const block<neon3> &b) {
    uint32x4_t a0 = b.left.a;
    uint32x4_t a1 = b.right.a;
    uint32x4_t a2 = b.left.b;
    uint32x4_t a3 = b.right.b;
    uint32x4_t a4 = b.left.c;
    uint32x4_t a5 = b.right.c;

    uint32x4_t b0 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a0)));
    uint32x4_t b1 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a1)));
    uint32x4_t b2 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a2)));
    uint32x4_t b3 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a3)));
    uint32x4_t b4 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a4)));
    uint32x4_t b5 = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(a5)));

    size_t d = 8 - (12 - S) * 2;
    uint32_t *q = reinterpret_cast<uint32_t *>(p);
    uint32x4x2_t e0 = { b0, b1 };
    uint32x4x2_t e1 = { b2, b3 };
    uint32x4x2_t e2 = { b4, b5 };
    vst2q_u32(q + d,     e1);
    vst2q_u32(q + d + 8, e2);
    vst2q_u32(q,         e0);
  }

  static inline std::pair<block<neon3>, cbc_state> cbc_post_decrypt(const block<neon3> &d, const block<neon3> &c, const cbc_state &state) {
    uint32x4_t c0 = c.left.a; // 3 2 1 0
    uint32x4_t c1 = c.right.a;
    uint32x4_t c2 = c.left.b; // d c b a
    uint32x4_t c3 = c.right.b;
    uint32x4_t c4 = c.left.c; // h g f e
    uint32x4_t c5 = c.right.c;

    uint32_t s4 = vgetq_lane_u32(c4, 3); // h
    uint32_t s5 = vgetq_lane_u32(c5, 3);

    uint32x4_t b0 = vextq_u32(c0, c0, 3); // 2 1 0 3
    uint32x4_t b1 = vextq_u32(c1, c1, 3);
    uint32x4_t b2 = vextq_u32(c0, c2, 3); // c b a 3
    uint32x4_t b3 = vextq_u32(c1, c3, 3);
    uint32x4_t b4 = vextq_u32(c2, c4, 3); // g f e d
    uint32x4_t b5 = vextq_u32(c3, c5, 3);

    uint32x4_t x0 = vsetq_lane_u32(state.left,  b0, 0); // 2 1 0 s
    uint32x4_t x1 = vsetq_lane_u32(state.right, b1, 0);
    uint32x4_t x2 = b2;                                 // c b a 3
    uint32x4_t x3 = b3;
    uint32x4_t x4 = b4;                                 // g f e d
    uint32x4_t x5 = b5;

    uint32x4_t d0 = d.left.a;  // 3 2 1 0
    uint32x4_t d1 = d.right.a;
    uint32x4_t d2 = d.left.b;  // d c b a
    uint32x4_t d3 = d.right.b;
    uint32x4_t d4 = d.left.c;  // g f e d
    uint32x4_t d5 = d.right.c;

    uint32x4_t p0 = veorq_u32(d0, x0); // 3 2 1 0
    uint32x4_t p1 = veorq_u32(d1, x1);
    uint32x4_t p2 = veorq_u32(d2, x2); // d c b a?
    uint32x4_t p3 = veorq_u32(d3, x3);
    uint32x4_t p4 = veorq_u32(d4, x4); // h g f e
    uint32x4_t p5 = veorq_u32(d5, x5);

    return std::make_pair(block<neon3>(neon3(p0, p2, p4), neon3(p1, p3, p5)), cbc_state(s4, s5));
  }

  static inline block<neon3> decrypt(const block<neon3> &b, const work_key_type &wk, int n) {
    /*
     * Register map
     * a: |  a.l  |  a.r  |   q2  |   q3  |
     * b: |  b.l  |  b.r  |   q4  |   q5  |
     * c: |  c.l  |  c.r  |   q6  |   q7  |
     * k: | 1 |k01|k23|k45|   k6  |   k7  |
     */
    block<neon3> t = b;

    uint32x4_t k7 = vdupq_n_u32(wk[7]);
    uint32x4_t k6 = vdupq_n_u32(wk[6]);
    uint32x2_t k45 = { wk[4], wk[5] };
    uint32x2_t k23 = { wk[2], wk[3] };
    uint32x2_t k01 = { wk[0], wk[1] };
    uint16x4_t one = vdup_n_u16(1);

    for (int i = 0; i < n; ++i) {
      __asm__ (
        // pi4
        "vadd.u32       q2, %q[ar], %q[k7];"
                                        "vadd.u32       q4, %q[br], %q[k7];"
                                                                        "vadd.u32       q6, %q[cr], %q[k7];"
        "vshr.u32       q3, q2, #30;"
                                        "vshr.u32       q5, q4, #30;"
                                                                        "vshr.u32       q7, q6, #30;"
        "vsli.u32       q3, q2, #2;"
                                        "vsli.u32       q5, q4, #2;"
                                                                        "vsli.u32       q7, q6, #2;"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q5;"
                                                                        "vadd.u32       q6, q7;"
        "vaddw.u16      q2, %[one];"
                                        "vaddw.u16      q4, %[one];"
                                                                        "vaddw.u16      q6, %[one];"
        "veor           %q[al], q2;"
                                        "veor           %q[bl], q4;"
                                                                        "veor           %q[cl], q6;"
        // pi3
    "vdup.u32       q3, %[k45][1];"
        "vadd.u32       q2, %q[al], q3;"
                                        "vadd.u32       q4, %q[bl], q3;"
                                                                        "vadd.u32       q6, %q[cl], q3;"
        "vshr.u32       q3, q2, #30;"
                                        "vshr.u32       q5, q4, #30;"
                                                                        "vshr.u32       q7, q6, #30;"
        "vsli.u32       q3, q2, #2;"
                                        "vsli.u32       q5, q4, #2;"
                                                                        "vsli.u32       q7, q6, #2;"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q5;"
                                                                        "vadd.u32       q6, q7;"
        "vaddw.u16      q2, %[one];"
                                        "vaddw.u16      q4, %[one];"
                                                                        "vaddw.u16      q6, %[one];"
        "vshr.u32       q3, q2, #24;"
                                        "vshr.u32       q5, q4, #24;"
                                                                        "vshr.u32       q7, q6, #24;"
        "vsli.u32       q3, q2, #8;"
                                        "vsli.u32       q5, q4, #8;"
                                                                        "vsli.u32       q7, q6, #8;"
        "veor           q2, q3;"
                                        "veor           q4, q5;"
                                                                        "veor           q6, q7;"
        "vadd.u32       q2, %q[k6];"
                                        "vadd.u32       q4, %q[k6];"
                                                                        "vadd.u32       q6, %q[k6];"
        "vsra.u32       q2, q2, #31;"
                                        "vsra.u32       q4, q4, #31;"
                                                                        "vsra.u32       q6, q6, #31;"
        "vrev32.u16     q3, q2;"
                                        "vrev32.u16     q5, q4;"
                                                                        "vrev32.u16     q7, q6;"
        "veor           %q[ar], q3;"
                                        "veor           %q[br], q5;"
                                                                        "veor           %q[cr], q7;"
        "vorr           q2, %q[al];"
                                        "vorr           q4, %q[bl];"
                                                                        "vorr           q6, %q[cl];"
        "veor           %q[ar], q2;"
                                        "veor           %q[br], q4;"
                                                                        "veor           %q[cr], q6;"
        // pi2
    "vdup.u32       q3, %[k45][0];"
        "vadd.u32       q2, %q[ar], q3;"
                                        "vadd.u32       q4, %q[br], q3;"
                                                                        "vadd.u32       q6, %q[cr], q3;"
        "vshr.u32       q3, q2, #31;"
                                        "vshr.u32       q5, q4, #31;"
                                                                        "vshr.u32       q7, q6, #31;"
        "vsli.u32       q3, q2, #1;"
                                        "vsli.u32       q5, q4, #1;"
                                                                        "vsli.u32       q7, q6, #1;"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q5;"
                                                                        "vadd.u32       q6, q7;"
        "vsubw.u16      q2, %[one];"
                                        "vsubw.u16      q4, %[one];"
                                                                        "vsubw.u16      q6, %[one];"
        "vshr.u32       q3, q2, #28;"
                                        "vshr.u32       q5, q4, #28;"
                                                                        "vshr.u32       q7, q6, #28;"
        "vsli.u32       q3, q2, #4;"
                                        "vsli.u32       q5, q4, #4;"
                                                                        "vsli.u32       q7, q6, #4;"
        "veor           %q[al], q3;"
                                        "veor           %q[bl], q5;"
                                                                        "veor           %q[cl], q7;"
        "veor           %q[al], q2;"
                                        "veor           %q[bl], q4;"
                                                                        "veor           %q[cl], q6;"
        // pi1
        "veor           %q[ar], %q[al];"
                                        "veor           %q[br], %q[bl];"
                                                                        "veor           %q[cr], %q[cl];"
        // pi4
    "vdup.u32       q3, %[k23][1];"
        "vadd.u32       q2, %q[ar], q3;"
                                        "vadd.u32       q4, %q[br], q3;"
                                                                        "vadd.u32       q6, %q[cr], q3;"
        "vshr.u32       q3, q2, #30;"
                                        "vshr.u32       q5, q4, #30;"
                                                                        "vshr.u32       q7, q6, #30;"
        "vsli.u32       q3, q2, #2;"
                                        "vsli.u32       q5, q4, #2;"
                                                                        "vsli.u32       q7, q6, #2;"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q5;"
                                                                        "vadd.u32       q6, q7;"
        "vaddw.u16      q2, %[one];"
                                        "vaddw.u16      q4, %[one];"
                                                                        "vaddw.u16      q6, %[one];"
        "veor           %q[al], q2;"
                                        "veor           %q[bl], q4;"
                                                                        "veor           %q[cl], q6;"
        // pi 3
    "vdup.u32       q3, %[k01][1];"
        "vadd.u32       q2, %q[al], q3;"
                                        "vadd.u32       q4, %q[bl], q3;"
                                                                        "vadd.u32       q6, %q[cl], q3;"
        "vshr.u32       q3, q2, #30;"
                                        "vshr.u32       q5, q4, #30;"
                                                                        "vshr.u32       q7, q6, #30;"
        "vsli.u32       q3, q2, #2;"
                                        "vsli.u32       q5, q4, #2;"
                                                                        "vsli.u32       q7, q6, #2;"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q5;"
                                                                        "vadd.u32       q6, q7;"
        "vaddw.u16      q2, %[one];"
                                        "vaddw.u16      q4, %[one];"
                                                                        "vaddw.u16      q6, %[one];"
        "vshr.u32       q3, q2, #24;"
                                        "vshr.u32       q5, q4, #24;"
                                                                        "vshr.u32       q7, q6, #24;"
        "vsli.u32       q3, q2, #8;"
                                        "vsli.u32       q5, q4, #8;"
                                                                        "vsli.u32       q7, q6, #8;"
        "veor           q2, q3;"
                                        "veor           q4, q5;"
                                                                        "veor           q6, q7;"
    "vdup.u32       q3, %[k23][0];"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q3;"
                                                                        "vadd.u32       q6, q3;"
        "vsra.u32       q2, q2, #31;"
                                        "vsra.u32       q4, q4, #31;"
                                                                        "vsra.u32       q6, q6, #31;"
        "vrev32.u16     q3, q2;"
                                        "vrev32.u16     q5, q4;"
                                                                        "vrev32.u16     q7, q6;"
        "vorr           q2, %q[al];"
                                        "vorr           q4, %q[bl];"
                                                                        "vorr           q6, %q[cl];"
        "veor           %q[ar], q3;"
                                        "veor           %q[br], q5;"
                                                                        "veor           %q[cr], q7;"
        "veor           %q[ar], q2;"
                                        "veor           %q[br], q4;"
                                                                        "veor           %q[cr], q6;"
    "vdup.u32       q3, %[k01][0];"
        // pi2
        "vadd.u32       q2, %q[ar], q3;"
                                        "vadd.u32       q4, %q[br], q3;"
                                                                        "vadd.u32       q6, %q[cr], q3;"
        "vshr.u32       q3, q2, #31;"
                                        "vshr.u32       q5, q4, #31;"
                                                                        "vshr.u32       q7, q6, #31;"
        "vsli.u32       q3, q2, #1;"
                                        "vsli.u32       q5, q4, #1;"
                                                                        "vsli.u32       q7, q6, #1;"
        "vadd.u32       q2, q3;"
                                        "vadd.u32       q4, q5;"
                                                                        "vadd.u32       q6, q7;"
        "vsubw.u16      q2, %[one];"
                                        "vsubw.u16      q4, %[one];"
                                                                        "vsubw.u16      q6, %[one];"
        "vshr.u32       q3, q2, #28;"
                                        "vshr.u32       q5, q4, #28;"
                                                                        "vshr.u32       q7, q6, #28;"
        "vsli.u32       q3, q2, #4;"
                                        "vsli.u32       q5, q4, #4;"
                                                                        "vsli.u32       q7, q6, #4;"
        "veor           %q[al], q3;"
                                        "veor           %q[bl], q5;"
                                                                        "veor           %q[cl], q7;"
        "veor           %q[al], q2;"
                                        "veor           %q[bl], q4;"
                                                                        "veor           %q[cl], q6;"
        // pi1
        "veor           %q[ar], %q[al];"
                                        "veor           %q[br], %q[bl];"
                                                                        "veor           %q[cr], %q[cl];"

        : [al] "+w"(t.left.a), [ar] "+w"(t.right.a),
          [bl] "+w"(t.left.b), [br] "+w"(t.right.b),
          [cl] "+w"(t.left.c), [cr] "+w"(t.right.c)
        : [one] "w"(one), [k7] "w"(k7), [k6] "w"(k6), [k45]"w"(k45), [k23]"w"(k23), [k01]"w"(k01)
        : "q2", "q3", "q4", "q5", "q6", "q7");
    }

    return t;
  }

};

}

template<>
inline size_t block_size<arm::neon3<11> >() {
  return 88;
}

template<>
inline size_t block_size<arm::neon3<12> >() {
  return 96;
}

template<>
inline void block<arm::neon3<11> >::load(const uint8_t *p) {
  arm::neon3<11>::load_block(*this, p);
}

template<>
inline void block<arm::neon3<12> >::load(const uint8_t *p) {
  arm::neon3<12>::load_block(*this, p);
}

template<>
inline void block<arm::neon3<11> >::store(uint8_t *p) const {
  arm::neon3<11>::store_block(p, *this);
}

template<>
inline void block<arm::neon3<12> >::store(uint8_t *p) const {
  arm::neon3<12>::store_block(p, *this);
}

template<>
inline std::pair<block<arm::neon3<11> >, cbc_state> block<arm::neon3<11> >::cbc_post_decrypt(const block<arm::neon3<11> > &c, const cbc_state &state) const {
  return arm::neon3<11>::cbc_post_decrypt(*this, c, state);
}

template<>
inline std::pair<block<arm::neon3<12> >, cbc_state> block<arm::neon3<12> >::cbc_post_decrypt(const block<arm::neon3<12> > &c, const cbc_state &state) const {
  return arm::neon3<12>::cbc_post_decrypt(*this, c, state);
}


template<>
inline block<arm::neon3<11> > cipher<arm::neon3<11> >::decrypt(const block<arm::neon3<11> > &b, const work_key_type &wk, int n) {
  return arm::neon3<11>::decrypt(b, wk, n);;
}

template<>
inline block<arm::neon3<12> > cipher<arm::neon3<12> >::decrypt(const block<arm::neon3<12> > &b, const work_key_type &wk, int n) {
  return arm::neon3<12>::decrypt(b, wk, n);;
}


}
