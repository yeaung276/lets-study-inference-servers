## Introduction

Ever wonder how those big language models are served across millions of requests on a cluster of GPU? Walk me throught a deep dive on how those inference engine works and how you would create one yourself.&#x20;

Today I am starting the series of blog posts where I dive down on most of the inference engines like vLLM, deepseek infer, to study the inner working of each engine look like and understand the concept. After that we are gonna create our own small inference engine from scretch using rust.&#x20;

Lets get started.



## The framework

To understand all of these inference servers works, we need to have a framework we need to tackle first. (Todo: actually research the framework, now, I am having my own, if ther eis something already a model of frameworks, comapre and rewriete. if nothing happen, move on). The will help us speed up moving a lot of those thousands and thousands of line code.

### Dimension 1: engines that optimize the GPU utilization and sheer throughput, or engines that optimize single user usecase and optimize efficiency and edge devices.

### Dimension 2: How are their memory layout and underlying kernals are differ(how they handle KVCache, how they optimize compute intensity).&#x20;

### Dimension 3: How they handl distributed load configurations, TP, DP, PP
