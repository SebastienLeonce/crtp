#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

// Implémentation basique d'expression templates:
// arborescence d'expressions à base de CRTP

// Voir:
// https://en.wikipedia.org/wiki/Expression_templates
// https://bitbucket.org/blaze-lib/blaze

namespace et {

// dense_expr_t est une classe-mère qui implémente l'interface
// d'une expression dense, ie. une expression dont le type implémente
// un operator[] et une méthode size().

template <typename T> struct dense_expr_t {
  // operator~: permet de voir un dense_expr_t comme son type sous-jacent
  T &operator~() { return *(T *)(this); }
  T const &operator~() const { return *(T const *)(this); }

  // On attend d'un dense_expr_t qu'il ait:
  // - un using element_t
  // - une méthode operator[](std::size_t i) const
  // - une méthode size() const qui renvoie la taille du conteneur

  // Optionnellement, on peut définir les méthodes vides en attente
  // de surcharge dans la classe fille.
  // Objectif: que le compilo *plante* si les méthodes ne sont pas surchargées
  // car on ne veut pas qu'il accepte du code pourri:

  using element_t = void;
  void operator[](std::size_t i) const {}
  void size() const {}

  // NB: Étant donné que le polymorphisme est *statique* et non dynamique
  // (contrairement à l'héritage virtuel), les signatures des fonctions peuvent
  // différer entre la classe mère et la classe fille, puisqu'elles sont
  // résolvables lors de la compilation.
};

// vector_t implémente une classe vector qui repose sur std::vector et
// surcharge la classe dense_expr_t
template <typename T> struct vector_t : dense_expr_t<vector_t<T>> {
  // Surcharge
  using element_t = T;

private:
  std::vector<T> v_;

public:
  // Constructeurs à la std::vector
  vector_t() : v_() {}
  vector_t(std::vector<T> v) : v_(v) {}
  vector_t(std::size_t s) : v_(s) {}
  vector_t(std::size_t s, element_t const &v) : v_(s, v) {}

  // Operator[] permettant de *modifier* le conteneur,
  // non-requis par l'interface dense_expr_t
  T &operator[](std::size_t i) { return v_[i]; }

  // Surcharges
  T const &operator[](std::size_t i) const { return v_[i]; }
  std::size_t size() const { return v_.size(); }

  // Implémentez l'opérateur d'affectation permettant d'évaluer une expression
  // directement dans un vector_t
  template <typename E> vector_t &operator=(dense_expr_t<E> const &e) {
    if ((*this).size() != (~e).size()) {
      std::exit(1);
    }

    for (size_t i = 0; i < (~e).size(); i++)
    {
      (*this)[i] = (~e)[i];
    }

    return *this;    
  }
};

// On donne monop_t qui implémente une expression dense qui a pour membres
// une fonction arbitraire et une référence vers une dense_expr_t arbitraire,
// et dont l'operator[](std::size_t i) renvoie l'élément du membre a[i]
// par la fonction f.

template <typename F, typename A> struct monop_t : dense_expr_t<monop_t<F, A>> {
  using f_t = F; // Type de la fonction
  using a_t = A; // Type du membre gauche

  // Surchargez element_t

  // Surcharge
  using element_t =
      std::decay_t<decltype(std::declval<f_t>()(std::declval<a_t>()[0]))>;

private:
  // Déclaration des membres
  f_t f_;
  a_t const &a_;

public:
  monop_t(F f, A const &a) : f_(std::move(f)), a_(a) {}

  // Surcharge
  element_t operator[](std::size_t i) const { return f_(a_[i]); }
  std::size_t size() const { return a_.size(); }
};

// Implémentez binop_t qui implémente une expression dense qui a pour membres
// une fonction arbitraire et deux références vers des dense_expr_t arbitraires,
// et dont l'operator[](std::size_t i) renvoie f(a[i], b[i]).

template <typename F, typename A, typename B> struct binop_t : dense_expr_t<binop_t<F, A, B>> {
  // Définir correctement les usings:

  using f_t = F; // Type de la fonction
  using a_t = A; // Type du membre gauche
  using b_t = B; // Type du membre droit

  // Surcharge

  using element_t = std::decay_t<decltype(std::declval<f_t>()(std::declval<a_t>()[0], std::declval<b_t>()[0]))>;

private:
  // Déclaration des membres
  f_t f_;
  a_t const &a_;
  b_t const &b_;

public:
  // Complétez le constructeur:
  binop_t(F f, A const &a, B const &b) : f_(std::move(f)), a_(a), b_(b) {
    if (a_.size() != b_.size()) {
      std::exit(1);
    }
  }

  // Implémentez l'operator[] et la méthode size():
  element_t operator[](std::size_t i) const { return f_(a_[i], b_[i]); }
  std::size_t size() const { return a_.size(); }
};

// Implémentez add_t qui représente une addition sur deux dense_expr:
template <typename A, typename B> struct add_t : dense_expr_t<add_t<A, B>> {
  private:
    A const &a_;
    B const &b_;

  public: 
    add_t(A const &a, B const &b) : a_(a), b_(b) {
      if (a.size() != b.size()) {
        std::exit(1);
      }
    }

    using element_t = std::decay_t<decltype(std::declval<A>()[0] + std::declval<B>()[0])>;

    element_t operator[](std::size_t i) const { return a_[i] + b_[i]; }
    std::size_t size() const { return a_.size(); }
};

// Implémentez les surcharges d'opérateurs qui permettent de creer des
// expressions pour le noeud add sur des dense_expr
template <typename E1, typename E2>
auto operator+(dense_expr_t<E1> const &l, dense_expr_t<E2> const &r) {
  return add_t<E1, E2>(~l, ~r);
}

// Implémentez de manière similaire une expression mul et les opérateurs
// associés
template <typename A, typename B> struct mul_t : dense_expr_t<mul_t<A, B>> {
  private:
    A const &a_;
    B const &b_;

  public: 
    mul_t(A const &a, B const &b) : a_(a), b_(b) {
      if (a.size() != b.size()) {
        std::exit(1);
      }
    }

    using element_t = std::decay_t<decltype(std::declval<A>()[0] * std::declval<B>()[0])>;

    element_t operator[](std::size_t i) const { return a_[i] * b_[i]; }
    std::size_t size() const { return a_.size(); }
};

template <typename E1, typename E2>
auto operator*(dense_expr_t<E1> const &l, dense_expr_t<E2> const &r) {
  return mul_t<E1, E2>(~l, ~r);
}
// [[BONUS]]
// Implémentez de manière similaire les expressions et opérateurs pour - / %
template <typename A, typename B> struct sub_t : dense_expr_t<sub_t<A, B>> {
  private:
    A const &a_;
    B const &b_;

  public: 
    sub_t(A const &a, B const &b) : a_(a), b_(b) {
      if (a.size() != b.size()) {
        std::exit(1);
      }
    }

    using element_t = std::decay_t<decltype(std::declval<A>()[0] - std::declval<B>()[0])>;

    element_t operator[](std::size_t i) const { return a_[i] - b_[i]; }
    std::size_t size() const { return a_.size(); }
};

template <typename E1, typename E2>
auto operator-(dense_expr_t<E1> const &l, dense_expr_t<E2> const &r) {
  return sub_t<E1, E2>(~l, ~r);
}

template <typename A, typename B> struct div_t : dense_expr_t<div_t<A, B>> {
  private:
    A const &a_;
    B const &b_;

  public: 
    div_t(A const &a, B const &b) : a_(a), b_(b) {
      if (a.size() != b.size()) {
        std::exit(1);
      }
    }

    using element_t = std::decay_t<decltype(std::declval<A>()[0] / std::declval<B>()[0])>;

    element_t operator[](std::size_t i) const { return a_[i] / b_[i]; }
    std::size_t size() const { return a_.size(); }
};

template <typename E1, typename E2>
auto operator/(dense_expr_t<E1> const &l, dense_expr_t<E2> const &r) {
  return div_t<E1, E2>(~l, ~r);
}

template <typename A, typename B> struct modulo_t : dense_expr_t<modulo_t<A, B>> {
  private:
    A const &a_;
    B const &b_;

  public: 
    modulo_t(A const &a, B const &b) : a_(a), b_(b) {
      if (a.size() != b.size()) {
        std::exit(1);
      }
    }

    using element_t = std::decay_t<decltype(std::declval<A>()[0] % std::declval<B>()[0])>;

    element_t operator[](std::size_t i) const { return a_[i] % b_[i]; }
    std::size_t size() const { return a_.size(); }
};

template <typename E1, typename E2>
auto operator%(dense_expr_t<E1> const &l, dense_expr_t<E2> const &r) {
  return modulo_t<E1, E2>(~l, ~r);
}

// BONUS 2: Implementez l'expression div_constante_t qui divise par une
// constante en multipliant par l'inverse de cette constante

template <typename A> struct div_constante_t : dense_expr_t<div_constante_t<A>>  {
  private:
    A const &a_;
    float const b_;

  public: 
    div_constante_t(A const &a, int const b) : a_(a), b_(1.f/b) {}

    using element_t = std::decay_t<decltype(std::declval<A>()[0] * std::declval<float>())>;

    element_t operator[](std::size_t i) const { return a_[i] * b_; }
    std::size_t size() const { return a_.size(); }
};

template <typename E1>
auto operator/(dense_expr_t<E1> const &l, int const r) {
  return div_constante_t<E1>(~l, r);
}
// Complétez l'implémentation de sum:

template <typename T> auto sum(T const &s) -> typename T::element_t {
  // On récupère le type des éléments
  using elmt_t = std::decay_t<decltype(std::declval<T>()[0])>;;

  // On récupère la taille de l'expression dense avec
  // la méthode size() *du type sous-jacent*
  auto const sz = (~s).size();

  // Initialisation de la valeur de retour
  elmt_t ret(0);

  // Calcul de la somme
  for (std::size_t i = 0; i < sz; i++)
    ret += (~s)[i];

  return ret;
}

// Implémentez des surcharges de sum() spécifiques à vector_t et add_t
// qui émettent respectivement "*vector_t* " et "*add_t* " dans la console
// avant de calculer et renvoyer la somme des éléments des expressions.

// Implémentez mean qui renvoie la moyenne d'une expression dense
// (sans se préoccuper de la conversion en flottant)
template <typename T> auto mean(T const &s) -> typename T::element_t {
  // On récupère le type des éléments
  using elmt_t = std::decay_t<decltype(std::declval<T>()[0])>;;

  // On récupère la taille de l'expression dense avec
  // la méthode size() *du type sous-jacent*
  auto const sz = (~s).size();

  // Initialisation de la valeur de retour
  elmt_t ret(0);

  // Calcul de la somme
  for (std::size_t i = 0; i < sz; i++)
    ret += (~s)[i];

  return ret/sz;
}
} // namespace et

int main() {
  constexpr std::size_t sz = 256;
  using v_t = et::vector_t<int>;

  v_t va(sz, 56), vb(sz, 42);
  et::binop_t op([](auto a, auto b) { return a / b; }, va, vb);

  std::cout << "op.size() = " << op.size() << '\n'
            << "op[23] = " << op[23] << '\n'
            << "(va * vb).size() = " << (va * vb).size() << '\n'
            << "(va * vb)[23] = " << (va * vb)[23] << '\n'
            << "(va / 10)[23] = " << (va / 10)[23] << '\n'
            << "sum(op) = " << sum(op) << '\n'
            << "sum(va) = " << sum(va) << '\n'
            << "mean(op) = " << mean(op) << '\n'
            << "mean(va) = " << mean(va) << '\n';
  //           << "mean(et::add_t(op, op)) = " << mean(et::add_t(op, op)) << '\n'
  //           << "mean(op + op) = " << mean(op + op) << '\n';

  // v_t vc;
  // vc = va + vb;
  // for (std::size_t i = 0; i < vc.size(); ++i)
  //   std::cout << vc[i] << " ";
  // std::cout << std::endl;
}
