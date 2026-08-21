---
title: 'Under the hood of Cairo 1.0: Exploring Sierra, Part 1'
pubDate: 2023-03-22
description: 'How Sierra makes Cairo programs safe before they become Cairo Assembly.'
related: ['cairo-1', 'cairo-1-sierra-2']
---

## About this article

This article was originally written [here](https://medium.com/nethermind-eth/under-the-hood-of-cairo-1-0-exploring-sierra-7f32808421f5) as part of my work at Nethermind.

## Part 1: An essential layer to produce safe CASM

## Introduction

I recently attended two conference talks at Starkware’s Sessions about Sierra: [Enforcing Safety Using Typesystems](https://www.youtube.com/watch?v=-EHwaQuPuAASierra) by Shahar Papini and [Not Stopping at the Halting Problem](https://www.youtube.com/watch?v=wYxUedcdVZ4) by Ori Ziv. I highly recommend you watch these videos if you want to understand more about the Cairo stack. The following post is the first of a series in which I’ll dive into Sierra to understand more about Cairo, its mechanisms, and Starknet overall.

Sierra (**S**afe **I**nt**e**rmediate **R**ep**r**esent**a**tion) is an intermediate layer between the high-level language Cairo and compilation targets such as Cairo Assembly (CASM). The language is designed to ensure safety and prevent runtime errors. It guarantees that every function returns and that there are no infinite loops by using a compiler that can detect operations that could fail at compile time. Sierra uses a simple yet powerful type system to express mid-level code while ensuring safety. This allows for efficient compilation down to CASM.

## Motivation

In Cairo 0, developers would write Starknet contracts in Cairo, compile them to CASM, and deploy the compilation output directly on Starknet. Users could then interact with Starknet contracts by calling smart contract functions, signing a transaction, and sending it to the sequencer. The sequencer would run the transaction to get the user’s transaction fee, the prover ([SHARP](https://www.cairo-lang.org/docs/sharp.html)) would generate the proof for the batch including this transaction, and the sequencer would charge a fee for including the transaction in a block.



*Cairo 0 transaction flow*

However, this Cairo 0 flow induces a few problems:

* Only valid statements can be proven in Cairo, so failed transactions cannot get proven. It is impossible to prove an invalid statement like `assert 0 = 1`, as it translates to polynomial constraints that can't be satisfied.
* Transaction execution may fail, resulting in the transaction not being included in a block. In this case, the sequencer does free work. Since failed transactions don’t have valid proofs, they can’t be included, and there is no way to enforce fee transfer for the sequencer.
* The sequencer can be subjected to DDoS attacks with invalid transactions, causing them to work for nothing and not collect any fee for running these transactions.
* It is impossible to distinguish between censorship (when a sequencer purposefully decides not to include certain transactions) and invalid transactions since both types of transactions will not be included in blocks.

On Ethereum, all failed transactions are marked as **reverted** but are still included in blocks, allowing validators to collect transaction fees upon failure. To prevent malicious users from bombarding the network with invalid transactions and overwhelming the sequencers, halting the network as legitimate transactions cannot be processed, Starknet needs a similar system that allows sequencers to collect fees for failed transactions. To solve the issues raised above, the Starknet network needs to achieve two goals: completeness and soundness. Completeness ensures that transaction execution can always be proven, even when it is expected to fail. Soundness ensures that valid transactions are not rejected, preventing censorship.

Sierra is correct-by-construction, letting the sequencer charge a fee for all transactions. Instead of deploying code that can fail (e.g., *asserts)*, we can deploy branching code (e.g., *if/else*). Cairo 1’s asserts are translated to branching Sierra code, allowing errors to be propagated back to the original entry point that returns a boolean value of *true/false* for transaction success or failure. The Starknet OS can determine if a transaction is valid based on the entry point return value, and decide whether or not to apply the state update if the transaction was successful.

Cairo 1 offers a syntax similar to Rust and abstracts the safety constructs of Sierra to create a provable, developer-friendly programming language. It compiles to Sierra, a correct-by-construction intermediate representation of Cairo code that doesn’t contain any failing semantics. This ensures that no Sierra code can ever fail, and it compiles later down to a safe subset of CASM. Developers can focus on writing efficient smart contracts without worrying about writing non-failing code, all with improved security primitives.

Instead of deploying CASM code to Starknet, developers will compile their Cairo 1 code to Sierra and deploy Sierra programs to Starknet. On a declare-transaction, the sequencers will be responsible for compiling the Sierra code to CASM, ensuring that no failing code can be deployed on Starknet.

## Correct by Construction

To design a language that cannot fail, we must first identify the **unsafe** operations in Cairo 0. These include:

* Illegal memory address references; attempting to access an unallocated memory cell
* Assertions, since they can fail with no way to recover
* Multiple writes to the same memory address due to Cairo’s write-once memory model
* Infinite loops which make it impossible to determine whether a program will exit

## Ensuring that dereferences cannot fail

In Cairo 0, developers could write the following code that tried to access the content of an unallocated memory cell.

```
let (ptr:felt*) = alloc();
tempvar x = [ptr];
```

Sierra’s type system prevents common pointer-related errors by enforcing strict ownership rules and making use of smart pointers like `Box`**,** thereby enabling the detection and prevention of invalid pointer dereferencing at compile time. The `Box<T>` type serves as a pointer to a valid and initialized pointer instance and provides two functions for instantiating and dereferencing: `box_new<T>()` and `box_deref()`. By using the type system to catch dereferencing errors at compile time, CASM compiled from Sierra avoids invalid pointer dereferencing.

## Ensuring that no memory cell is being written twice

In Cairo 0, users would use arrays like this:

```
let (array:felt*) = alloc();
assert array[0] = 1;
assert array[1] = 2;
assert array[1] = 3;
```

However, attempting to write twice to the same array index results in a runtime error as memory cells can only be written once. To avoid this issue, Sierra introduces an `Array<T>` type along with an `array_append<T>(Array<T>, value:T) -> Array<T>` function. This function takes an instance of an array and a value to append and returns the updated array instance pointing to the new next free memory cell. As a result, values are sequentially appended to the end of arrays without worrying about possible clashes due to already-written memory cells.

## Non-failing assertions

Assertions are usually employed to assess the result of a boolean expression at a particular point in the code. If the evaluation is not as anticipated, an error is raised. Unlike in Cairo 0, the compilation of a Cairo 1 `assert` instruction will produce branching Sierra code. This code will terminate the current function execution prematurely if the assertion is not satisfied and continue with the next instructions otherwise.

## Ensuring soundness of programs using Dictionaries

Dictionaries suffer from the same multiple append problem as Arrays, which can be solved by introducing a special `Dict<K,V>` type and a set of utility functions to instantiate, retrieve, and set dictionary values. However, there is a soundness \*\*issue with dictionaries. Every Dict must be **squashed** at the end of a program by calling the `dict_squash(Dict<K,V>) -> ()` function, which verifies the consistency of the sequence of key updates. Unsquashed dictionaries are dangerous, as a malicious prover could prove the correctness of inconsistent updates.

As we previously saw, the linear type system enforces objects to be used exactly once. The only way to “use” a Dict is to call the `dict_squash` function, which uses the dictionaries instance and returns nothing. This means that unsquashed dictionaries will be detected when compiling Sierra code to CASM, and an error will be thrown at compile time. For other types that don't necessarily need to be used exactly once, usually types that don't contain Dicts, Sierra introduces the `drop<T>(T)->()` function that uses an instance of an object and doesn't return anything.

It’s worth noting that both `drop` and `dup` don't produce any CASM code. They simply provide type safety at the Sierra level, ensuring that variables are used exactly once.

## Preventing <mark>infinite</mark> loops

Determining whether a program will eventually halt or run forever is a fundamental problem in computer science known as the Halting Problem, and unfortunately, it is unsolvable in the general case. In a distributed environment like Starknet, where users can deploy and run arbitrary code, it is important to prevent users from running infinite loops such as the following Cairo code.

```
fn foo() {
	foo()
}
```

As recursive functions can be a source of infinite loops if the stop condition is never met, the Cairo-to-Sierra compiler will inject a `withdraw_gas` method at the beginning of recursive functions. As this feature has not yet been implemented, developers are still expected to call `withdraw_gas` in recursive functions and handle the result themselves, although it should be included in a future release of the compiler.

This `withdraw_gas` function will deduct the amount of gas required to run the function from the total gas available for the transaction by evaluating the cost of running each instruction in the function. The cost is analyzed by determining how many steps each operation takes - which is known at compile time for most operations. During the execution of a Cairo program, if the `withdraw_gas` call returns a null or negative value, the current function execution will stop, all pending variables will be consumed by calling `dict_squash` on unsquashed dictionaries and calling `drop` on other variables, and the execution will be considered as failed. Because all transactions on Starknet have a limited amount of spendable gas to execute the transactions, infinite loops are avoided - and by ensuring that enough gas is still available to drop the variables and stop the execution, the sequencer will be able to collect the fees from the transaction failure.

## Safe CASM achieved through a restricted set of instructions

The main goal of Sierra is to ensure that the generated CASM code cannot fail. To achieve this, Sierra programs are composed of statements that invoke `libfuncs`. These are a set of built-in library functions for which the generated CASM code is guaranteed to be safe. For example, the `array_append` libfunc generates safe CASM code to append a value to an array.

This approach to achieving code safety by only allowing a set of safe and trusted library functions is similar to the philosophy of the Rust programming language. By providing developers with a set of safe and trusted abstractions, both languages help to prevent common programming errors and increase the safety and reliability of the code. Cairo 1 uses an ownership and borrowing system similar to Rust’s, providing developers with a way to reason about code safety at compile-time, which can help to prevent bugs and improve overall code quality.

## TL;DR

Sierra serves as a crucial intermediary between the higher-level Cairo programming languages and more primitive compilation targets such as CASM by ensuring that the generated CASM is safe to run on Starknet. Its design is centered on safety, using a restricted set of functions that produce safe CASM code, a robust compiler combined with a linear type system to prevent runtime errors, and a built-in gas system to prevent infinite loops.

In the next part, we will focus on understanding the anatomy of a Sierra program, providing the basic requirements needed to read and understand Sierra programs.
