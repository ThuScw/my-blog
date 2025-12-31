---
title: "浅析LTE引理"
date: 2025-12-31T10:00:00+08:00
draft: false
categories: ["数学"]
tags: [ "数学","高中数学","数论"]
---

## 1.定理的描述
**定义**：$v_p(x)$表示素数$p$整除$x$的最高次幂。记$v_p(x)=\alpha$也即有$p^{\alpha}||x$

**引理**：$n\in \mathbb{Z}^+, x,y\in \mathbb{Z}, \forall p\in\mathbb{P},(p,n)=1,p|x-y,p\nmid x,p\nmid y$，有
$v_p(x^n-y^n)=v_p(x-y),v_p(x^n+y^n)=v_p(x+y)$

**定理1**：$n\in \mathbb{Z}^+, x,y\in \mathbb{Z}, \forall p\in\mathbb{P},p|x-y,p\nmid x,p\nmid y$，有$v_p(x^n-y^n)=v_p(x-y)+v_p(n),v_p(x^n+y^n)=v_p(x+y)+v_p(n)$

**定理2（LTE中 $n=2$情形1）**：$n\in \mathbb{Z}^+, 4|x-y,2\nmid x,2\nmid y$，有$v_2(x^n-y^n)=v_2(x-y)+v_2(n)$

**定理3（LTE中 $n=2$情形2）**：$n\in \mathbb{Z}^+, 2|n,2\nmid x,2\nmid y$，有$v_2(x^n-y^n)=v_2(x-y)+v_2(x+y)+v_2(n)-1$



---

证明：不会，我菜。参考[这里](https://zhuanlan.zhihu.com/p/106900309)

---
## 2.基本性质和用法

$v_p(x)+v_p(y)=v_p(xy)$

$v_p(x)-v_p(y)=v_p(\frac{x}{y})$（此时需要$y|x$)

是不是很像$\log$函数

---

我们可以知道，$x|y \Leftrightarrow\forall p|x,p\in\mathbb{P},v_p(x)\leqslant v_p(y)$

---

$eg1.$求证$2^{m+3}|3^{2^{m+2}}-1$

应用引理，有$v_2(3^{2^{m+2}}=v_2(3-1)+v_2(3+1)+v_2(2^{m+2})-1=m+4 > v_2(2^{m+3})=m+3$知其成立

---

$eg2.$已知$a,b,n\in\mathbb{N}^+$,满足$n|a^n-b^n$，求证$n|\frac{a^n-b^n}{a-b}$

考虑n的一个素因子$p$

若 $(p,a-b)=1$，有$v_p(\frac{a^n-b^n}{a-b})=v_p(a^n+b^n)\geqslant v_p(n)$

若$p|a,p|a$，有$p^{v_p(\frac{a^n-b^n}{a-b})}\geqslant p^{n-1}\geqslant n\geqslant p^{v_p(n)}$

若$p|a-b,(p,ab)=1$,有$v_p(\frac{a^n-b^n}{a-b})=v_p(n)$

---

## 3.具体做题

利用LTE引理在各种数论题目里乱搞

$eg1.\;$求$x^{2022}+y^{2022}=337^z$的全部正整数解

由费马小定理$337|(x^{2022}+y^{2022})-(x+y)$，故$337|x+y$

故$v_{337}(x^{2022}+y^{2022})=v_{337}(2022)+v_{337}(x+y)=v_{337}(337(x+y))$

因此$x^{2022}+y^{2022}\leqslant 337^{v_{337}(x^{2022}+y^{2022})}=337^{v_{337}(337(x+y))}\leqslant 337(x+y)$

（显然$p^{v_p(x)}≤x$）

而不等式$x^{2022}+y^{2022}\leqslant 337(x+y)$在$x,y$不全等于1时显然不成立

而且$x=y=1$也不成立

因此原方程无正整数解

---
$eg2.\;m\in\mathbb{Z},p\in\mathbb{P},$令$a_1=8p^m,a_n=(n+1)^{\frac{a_{n-1}}{n}}$，求所有素数$p$使得对于一切$n\in\mathbb{N}^+$，均有$a_n\prod\limits_{i=1}^n(1-\frac{1}{a_i})\in\mathbb{Z}$

由于对一切n成立，不妨设$n=2$，此时$a_n\prod\limits_{i=1}^n(1-\frac{1}{a_i})=\frac{(a_1-1)(a_2-1)}{a_1}\in\mathbb{Z}$

$\Rightarrow a_1|a_2-1\Leftrightarrow 8p^m|3^{4p^m}-1\Rightarrow p|3^{4p^m}-1$

$3^{4p^m}-1\equiv3^4-1\equiv80\equiv0\pmod{p}\Rightarrow p=2\,\,\,\,or\,\,\,5$

当$p=2$时，只需证$2^{m+3}|3^{2^{m+2}}-1$

由LTE引理知，$v_2(3^{2^{m+2}}-1)=v_2(3+1)+v_2(3-1)+v_2(2^{m+2})-1=m+4>v_2(2^{m+3})=m+3$，所以上式成立。

当$p=5$时，只需证$8·5^m|3^{4·5^m}-1$，由于$8|3^{4·5^m}-1$，只需证$5^m|3^{4·5^m}-1$

由LTE引理知，$v_5(3^{4·5^m}-1)=v_5(3^4-1)+v_5(5^m)=m+1>v_5(5^m)=m$，所以上式成立。

所以可以知道$p=2$或$5$

考虑一般情况，$a_n\prod\limits_{i=1}^n(1-\frac{1}{a_i})\in\mathbb{Z}\Leftrightarrow \frac{(a_1-1)·……·(a_n-1)}{a_1a_2……a_{n-1}}\in\mathbb{Z}$

猜测有$\forall n>2,a_{n-1}|a_n-1$,下证之

任取$a_{n-1}$的素因子p，只需证$v_p((n+1)^{\frac{a_{n-1}}{n}}-1)\geqslant v_p(a_{n-1})$

$a_n=(n+1)^{\frac{a_{n-1}}{n}}\Rightarrow \frac{a_n}{n+1}=(n+1)^{\frac{a^{n-1}}{n}-1}\Rightarrow$数归易证，$\frac{a_{n-1}}{n}\in\mathbb{N}^+$

也即$a_n$为$n+1$的正整数次幂，记$a_n=(n+1)^{\alpha_n}$,由LTE知，$v_n((n+1)^{\frac{a_{n-1}}{n}}-1)=v_n(n+1-1)+v_n(\frac{a_{n-1}}{n})=\alpha_{n-1}=v_n(a_{n-1})$得证。

综上，$p=2$或$5$

---
$E\ n\ d\ .$