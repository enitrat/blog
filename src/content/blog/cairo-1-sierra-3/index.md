---
title: 'Under the hood of Cairo 1.0: Exploring Sierra, Part 3'
pubDate: 2023-05-30
description: 'A practical guide to becoming a better Cairo developer with Sierra.'
related: ['cairo-1-sierra-2']
---

## About this article

This article was originally written [here](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-1220f6dbcf9) as part of my work at Nethermind.

## Part 3: Become a better Cairo developer with Sierra

## Introduction

In the [first blog post of the series](/writing/cairo-1-sierra-1/), we introduced Sierra, an intermediate language designed to simplify the development process of Starknet contracts by ensuring that all deployed code on Starknet cannot produce errors at runtime. [In the second post](/writing/cairo-1-sierra-2/), we analyzed the structure of a Sierra program to provide a comprehensive understanding of Sierra and why it is a safe intermediate representation of the Cairo code. In our closing article, we’ll delve deeper into some core and novel concepts introduced in Cairo 1 by analyzing Sierra code.

## Mutable variables, References, and Snapshots

Cairo introduces a new idea of snapshots which is often confused with references. There are three different ways of passing variables in function calls in Cairo: pass-by-value, where the caller function takes ownership of the variable; pass-by-reference, where the caller function "borrows" the variable and returns ownership to the caller context after its execution; and pass-by-snapshot, where you create a snapshot of a value, which is an immutable view to a value, and pass it to the function so that you keep ownership of the base value.

The Rust equivalents of `ref` and `@` (snapshots) would be `ref <=> &mut, @ <=> &`, but there are subtleties to be aware of. Special care must be put into understanding how mutable references can be passed as function parameters and how values are mutated. Snapshots are a concept exclusive to Cairo 1 and do not have a direct equivalent in other languages! Looking at Sierra code will give us a better understanding.

## Mutable variables

Let's start with a simple concept - mutable variables. In a traditional programming language, each variable is associated with a specific memory cell, a location in the computer's memory where the variable's data is stored. When a variable is assigned a value, the value is stored in the memory cell associated with that variable. The variable can then access or modify the value stored in that memory cell throughout the program's execution.

However, in Cairo, it's impossible to modify the content of a memory cell that has already been written to. Analyzing the compiled Sierra code, let's see what exactly happens when you declare a variable as mut in a Cairo program. Let's consider the Cairo program, in which a variable `x` is declared as `mut` and `y` is not, and we shadow the previous declaration of `y`.

```
fn main() {
    let mut x = 3;
    x = 5;

    let y = 30;
    let y = 50;
}
```

The Sierra code it compiles to is:

```
type felt = felt;
type Unit = Struct<ut@Tuple>;

libfunc felt_const<3> = felt_const<3>;
libfunc drop<felt> = drop<felt>;
libfunc felt_const<5> = felt_const<5>;
libfunc felt_const<30> = felt_const<30>;
libfunc felt_const<50> = felt_const<50>;
libfunc struct_construct<Unit> = struct_construct<Unit>;
libfunc store_temp<Unit> = store_temp<Unit>;

felt_const<3>() -> ([0]);
drop<felt>([0]) -> ();
felt_const<5>() -> ([1]);
drop<felt>([1]) -> ();
felt_const<30>() -> ([2]);
drop<felt>([2]) -> ();
felt_const<50>() -> ([3]);
drop<felt>([3]) -> ();
struct_construct<Unit>() -> ([4]);
store_temp<Unit>([4]) -> ([5]);
return([5]);

example::main@0() -> (Unit);
```

As demonstrated in this compiled Sierra program, mutable variables are syntactic sugar that enables Cairo developers to effortlessly modify and update data values throughout a program's execution without having to manually shadow the previously declared variable. When we modify our mutable variable `x`, the corresponding Sierra variable storing its value is first dropped as it's no longer used, and then a new variable is created with the updated value, as shown on lines 13-14. Similarly, for our non-mutable variable `y`, whose value is shadowed, the procedure in Sierra is exactly the same: the prior value is dropped, and a new one is instantiated with the updated value associated with `y`, as shown on lines 17-18. It is, however, recommended to use mutable variables instead of shadowing where possible, as it ensures consistency in types.

The `Unit` type declared represents an empty Struct and is the type returned by default by functions that don't return values.

## References

In traditional languages, “pass-by-reference” is a method of passing variables to functions where the function receives a reference to the variable's memory location. This allows the function to modify the variable's value directly. In Cairo, the equivalent is achieved using the `ref` modifier when defining the function parameter. However, as previously stated, it's essential to note that once assigned variable values can't be modified directly in Cairo, unlike in other languages.

Consider the following code snippet in Cairo:

```
fn main() -> felt {
    let mut x = 1;
    increment(ref x);
    x
}

fn increment(ref x: felt) {
    x+=1;
}
```

In this example, the `x` variable is defined as mutable using the `mut` keyword, and a mutable reference to `x` is passed to the `increment` function using the `ref` prefix. The function directly increments the value of `x`, and the new value is returned.

To further understand how this operates at a lower level, let's analyze the corresponding Sierra code:

```
type felt252 = felt252;
type Unit = Struct<ut@Tuple>;

libfunc felt252_const<1> = felt252_const<1>;
libfunc store_temp<felt252> = store_temp<felt252>;
libfunc function_call<user@pass_by_ref::pass_by_ref::increment> = function_call<user@pass_by_ref::pass_by_ref::increment>;
libfunc drop<Unit> = drop<Unit>;
libfunc felt252_add = felt252_add;
libfunc struct_construct<Unit> = struct_construct<Unit>;
libfunc store_temp<Unit> = store_temp<Unit>;

felt252_const<1>() -> ([0]);
store_temp<felt252>([0]) -> ([3]);
function_call<user@pass_by_ref::pass_by_ref::increment>([3]) -> ([1], [2]);
drop<Unit>([2]) -> ();
store_temp<felt252>([1]) -> ([4]);
return([4]);
felt252_const<1>() -> ([1]);
felt252_add([0], [1]) -> ([2]);
struct_construct<Unit>() -> ([3]);
store_temp<felt252>([2]) -> ([4]);
store_temp<Unit>([3]) -> ([5]);
return([4], [5]);

pass_by_ref::pass_by_ref::main@0() -> (felt252);
pass_by_ref::pass_by_ref::increment@6([0]: felt252) -> (felt252, Unit);
```

The first observation here is the signature of the `increment` function from its declaration in Cairo. As expected, the function returns the default `Unit` type for functions without return values, which is anticipated. However, it also returns a `felt252`. When function parameters are declared as `ref`, the compiler will generate code to automatically return the updated value of the argument passed to the function without the need to specify it in the higher-level code.

This is another example of how Cairo provides syntactic sugar to improve the developer experience. The above code is functionally equivalent to the following pass-by-value code, and they compile roughly the same Sierra code.

```
fn main() -> felt {
    let mut x = 1;
    increment(x)
}

fn increment(mut x: felt) -> felt {
    x += 1;
    x
}
```

```
type felt252 = felt252;

libfunc felt252_const<1> = felt252_const<1>;
libfunc store_temp<felt252> = store_temp<felt252>;
libfunc function_call<user@pass_by_value::pass_by_value::increment> = function_call<user@pass_by_value::pass_by_value::increment>;
libfunc rename<felt252> = rename<felt252>;
libfunc felt252_add = felt252_add;

felt252_const<1>() -> ([0]);
store_temp<felt252>([0]) -> ([2]);
function_call<user@pass_by_value::pass_by_value::increment>([2]) -> ([1]);
rename<felt252>([1]) -> ([3]);
return([3]);
felt252_const<1>() -> ([1]);
felt252_add([0], [1]) -> ([2]);
store_temp<felt252>([2]) -> ([3]);
return([3]);

pass_by_value::pass_by_value::main@0() -> (felt252);
pass_by_value::pass_by_value::increment@5([0]: felt252) -> (felt252);
```

## Snapshots

In the Cairo programming language, snapshots are introduced as a wrapper type that creates an immutable view of an object at a given time. Snapshots are useful when we need to perform on non-duplicable types like arrays. In runtime implementation, snapshots are zero-cost abstraction because of Cairo Assembly’s write-once memory model.

zing this program, we observe:

* The snapshot type doesn't exist in this Sierra program. Instead, the Sierra code only uses `felt252`. The `pass_by_snapshot` signature takes a `felt252` as a parameter, even though we specified in our Cairo program that it should take a snapshot as a parameter.
* The `snapshot_take` libfunc takes a `felt252` as input and returns two variables. Its signature is very similar to the `dup` libfunc.
* The desnap operator `*` doesn't generate any Sierra code

To understand more about what's happening here, let's dive into the compiler code. In the [cairo-lang-sierra](https://github.com/starkware-libs/cairo/blob/main/crates/cairo-lang-sierra/src/extensions/modules/snapshot.rs#L37) crate, we learn that a snapshot is just a wrapper around an object that ensures the original object is not modified. The `snapshot_take` libfunc only returns a snapshot to the type if the type *cannot be copied*. Duplicatable types are their own snapshot - as the snapshot itself is useless if we can duplicate the value. This concept of snapshots only exists at the Sierra level and makes the linear type system effective by ensuring that the object wrapped in a snapshot can't be modified.

But when do we find snapshots particularly useful? Specifically when working with non-duplicable types like Arrays. In the following code, a function `foo` takes as a parameter an Array `a`. A snapshot to this array is passed to two functions, and the array is then returned.

```
fn foo(a: Array::<felt252>) -> Array::<felt252> {
    bar(@a);
    bar_2(@a);
    a
}
fn bar(a: @Array::<felt252>) {}

fn bar_2(a: @Array::<felt252>) {}
```

```
type felt252 = felt252;
type Array<felt252> = Array<felt252>;
type Snapshot<Array<felt252>> = Snapshot<Array<felt252>>;
type Unit = Struct<ut@Tuple>;

libfunc snapshot_take<Array<felt252>> = snapshot_take<Array<felt252>>;
libfunc store_temp<Snapshot<Array<felt252>>> = store_temp<Snapshot<Array<felt252>>>;
libfunc function_call<user@snapshot_2::snapshot_2::bar> = function_call<user@snapshot_2::snapshot_2::bar>;
libfunc drop<Unit> = drop<Unit>;
libfunc function_call<user@snapshot_2::snapshot_2::bar_2> = function_call<user@snapshot_2::snapshot_2::bar_2>;
libfunc store_temp<Array<felt252>> = store_temp<Array<felt252>>;
libfunc drop<Snapshot<Array<felt252>>> = drop<Snapshot<Array<felt252>>>;
libfunc struct_construct<Unit> = struct_construct<Unit>;
libfunc store_temp<Unit> = store_temp<Unit>;

snapshot_take<Array<felt252>>([0]) -> ([1], [2]);
store_temp<Snapshot<Array<felt252>>>([2]) -> ([4]);
function_call<user@snapshot_2::snapshot_2::bar>([4]) -> ([3]);
drop<Unit>([3]) -> ();
snapshot_take<Array<felt252>>([1]) -> ([5], [6]);
store_temp<Snapshot<Array<felt252>>>([6]) -> ([8]);
function_call<user@snapshot_2::snapshot_2::bar_2>([8]) -> ([7]);
drop<Unit>([7]) -> ();
store_temp<Array<felt252>>([5]) -> ([9]);
return([9]);
drop<Snapshot<Array<felt252>>>([0]) -> ();
struct_construct<Unit>() -> ([1]);
store_temp<Unit>([1]) -> ([2]);
return([2]);
drop<Snapshot<Array<felt252>>>([0]) -> ();
struct_construct<Unit>() -> ([1]);
store_temp<Unit>([1]) -> ([2]);
return([2]);

snapshot_2::snapshot_2::foo@0([0]: Array<felt252>) -> (Array<felt252>);
snapshot_2::snapshot_2::bar@10([0]: Snapshot<Array<felt252>>) -> (Unit);
snapshot_2::snapshot_2::bar_2@14([0]: Snapshot<Array<felt252>>) -> (Unit);
```

In the generated Sierra code, we note the declaration a Snapshot type. Unlike our previous example, the `snapshot_take` libfunc returns both the original object and a snapshot of the original object - a wrapper type around our object. This snapshot is then passed to our functions. If you attempt to call a function that modifies the array object, such as the `array_append` libfunc, the Sierra program will not compile to CASM because a type mismatch will be detected at compile time. This is because you are attempting to append to a `Snapshot<Array<T>>` type, but the `array_append` libfunc expects an `Array<T> type`.

In summary, when a function takes a snapshot to a value using `@`, it is only able to read the value and not modify it. It behaves like an immutable borrow using `&` in Rust, which allows multiple parts of the program to read the same value simultaneously while ensuring that it is not modified. When working with non-copyable objects, using snapshots allows you to retain ownership of the object in the calling context while ensuring the object remains unaltered.

## Function inlining

Function inlining is a compiler optimization technique that substitutes a function call with the actual code of the function being called. It eliminates the overhead of a function call by integrating the function's code directly into the calling function.

The Cairo compiler will automatically replace calls to functions marked as inline directly with their Sierra code. This optimization is especially useful for frequently called small functions. Inlining can reduce the overhead of function calls and lead to faster and more optimised executions, as values don't need to be pushed to memory. Consider the Cairo program where the first function has an `[inline(always)]` attribute, while the second doesn't.

```
fn main() {
    inlined();
    not_inlined();
}

#[inline(always)]
fn inlined() -> felt {
    1 + 1
}

fn not_inlined() -> felt {
    2 + 2
}
```

```
type felt252 = felt252;
type Unit = Struct<ut@Tuple>;

libfunc felt252_const<1> = felt252_const<1>;
libfunc store_temp<felt252> = store_temp<felt252>;
libfunc felt252_add = felt252_add;
libfunc drop<felt252> = drop<felt252>;
libfunc function_call<user@inline::inline::not_inlined> = function_call<user@inline::inline::not_inlined>;
libfunc struct_construct<Unit> = struct_construct<Unit>;
libfunc store_temp<Unit> = store_temp<Unit>;
libfunc felt252_const<2> = felt252_const<2>;

felt252_const<1>() -> ([0]);
felt252_const<1>() -> ([1]);
store_temp<felt252>([0]) -> ([0]);
felt252_add([0], [1]) -> ([2]);
drop<felt252>([2]) -> ();
function_call<user@inline::inline::not_inlined>() -> ([3]);
drop<felt252>([3]) -> ();
struct_construct<Unit>() -> ([4]);
store_temp<Unit>([4]) -> ([5]);
return([5]);
felt252_const<1>() -> ([0]);
felt252_const<1>() -> ([1]);
store_temp<felt252>([0]) -> ([0]);
felt252_add([0], [1]) -> ([2]);
store_temp<felt252>([2]) -> ([3]);
return([3]);
felt252_const<2>() -> ([0]);
felt252_const<2>() -> ([1]);
store_temp<felt252>([0]) -> ([0]);
felt252_add([0], [1]) -> ([2]);
store_temp<felt252>([2]) -> ([3]);
return([3]);

inline::inline::main@0() -> (Unit);
inline::inline::inlined@10() -> (felt252);
inline::inline::not_inlined@16() -> (felt252);
```

In the Sierra code resulting from this program, instead of executing the inline function using a `function_call` libfunc to execute the `inline` function at line 15, the compiler integrates the code directly into the main function.

However, using function inlining can increase the overall program size due to code duplication for each inlined function call. Therefore, it is recommended to use inlining only for frequently called functions that have a limited number of instructions.

## Conclusion

In this post, we have explored some core concepts of Cairo 1, like mutable variables, references, and snapshots. We have seen how mutable variables in Cairo are equivalent to shadowed variables in Sierra and how references in Cairo use the `ref` prefix to pass variables and implicitly return them. Additionally, we have seen how snapshots in Cairo are a unique concept that allows developers to keep ownership of objects while ensuring that the original value remains unmodified. Finally, we explored how developers can use function inlining as an optimization technique.

Understanding the core concepts of the Cairo stack is essential to becoming a better developer in the Starknet ecosystem, and we hope that this series has provided you with valuable insights and knowledge to improve your skills. Keep Starknet Strange and Flourishing!
