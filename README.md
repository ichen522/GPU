# Custom Neural Network GPU Accelerator (NetFPGA)

![Verilog](https://img.shields.io/badge/Language-Verilog-blue.svg)
![Python](https://img.shields.io/badge/Language-Python-yellow.svg)
![FPGA](https://img.shields.io/badge/Platform-NetFPGA-orange.svg)
![Simulation](https://img.shields.io/badge/Tool-ModelSim-lightgrey.svg)

## 📌 Project Overview
[cite_start]This project implements a custom, standalone GPU processor built from scratch, optimized for Artificial Neural Network (ANN) acceleration on a resource-constrained NetFPGA platform[cite: 6, 8, 20]. [cite_start]Designed with a focus on Matrix Multiplication (GEMM) workloads, the core features a custom Instruction Set Architecture (ISA) and a fully pipelined Tensor Unit capable of executing 64-bit SIMD BFloat16 operations[cite: 12, 14, 28]. 

[cite_start]In addition to the hardware design (RTL), this project includes a custom software compiler toolchain that parses NVIDIA's `nvcc`-generated PTX assembly from CUDA kernels and translates it into the custom GPU machine code[cite: 7, 38].

---

## 🚀 Key Architectural Highlights

### 1. 64-bit SIMD Tensor Unit (BFloat16 MAC)
[cite_start]The heart of the GPU is a custom-designed execution unit optimized for Fused Multiply-Accumulate (MAC) operations[cite: 13, 14]. 
* [cite_start]**Data Packing:** Treats 64-bit registers as packed vectors of 4x BFloat16 elements, processing them in parallel within a single SIMD instruction[cite: 17, 18, 39].
* **3-Stage Pipeline Architecture:** To meet strict timing requirements (Setup/Hold time) and maximize throughput, the MAC core is heavily pipelined:
  * **Stage 1:** Mantissa multiplication (8x8-bit) and Exponent addition.
  * **Stage 2:** Exponent comparison, alignment, and addition/subtraction.
  * **Stage 3:** Normalization (via a custom 17-bit Leading Zero Counter / Priority Encoder) and Activation.
* [cite_start]**Hardware ReLU Integration:** The Rectified Linear Unit ($out[i]=max(0,in[i])$) is integrated directly into the final pipeline stage of the MAC core, saving extra execution cycles for ANN workloads[cite: 29, 55].

### 2. Software-Hardware Co-Design (CUDA to Custom Opcode)
[cite_start]Built a compiler pipeline to bridge high-level CUDA programming with the custom Verilog datapath[cite: 59].
* [cite_start]Uses `nvcc -ptx -arch=sm_80` to compile `.cu` kernels into PTX assembly[cite: 64].
* [cite_start]Developed a Python-based parser to translate PTX instructions into the custom GPU hex machine code, mapping CUDA's `threadIdx.x` concepts to the hardware thread ID control mechanisms[cite: 32, 33, 38, 64].

---

## 🧠 Hardware Microarchitecture (Datapath)

![Insert your High-Level Datapath Diagram Here]
[cite_start]*(Image: High-level block diagram depicting the Control Unit, Register File, LD/ST Unit, and 64-bit Tensor Unit datapath)* [cite: 69, 70]

* [cite_start]**Control Unit:** Manages the fetch-decode-execute cycle using a single program counter[cite: 30, 31]. Includes pipeline stall logic to handle Data Hazards (Read-After-Write) from the 3-cycle latency of the Tensor Unit.
* [cite_start]**Register File:** Supports 64-bit wide vector addressing, providing concurrent read ports to supply `A`, `B`, and `C` operands for the $A \times B + C$ operation[cite: 23, 24].
* [cite_start]**Memory Interface (LD/ST Unit):** Loads and stores full 64-bit packed registers from/to the Block RAM[cite: 35].

---

## 🛠️ Verification & Simulation

Rigorously verified the RTL design using **ModelSim**. 

![Insert your ModelSim Waveform Screenshot Here]
*(Image: ModelSim waveform demonstrating the 3-cycle pipeline latency and correct BFloat16 parallel computation, including successful hardware ReLU filtering for negative outputs.)*

* Validated parallel execution of 4x BFloat16 MAC operations across multiple lanes.
* Confirmed cycle-accurate timing and correct propagation of `valid` signals across pipeline stages.
* [cite_start]Python-based Instruction Set Simulator (ISS) built to pre-verify custom opcodes before RTL synthesis[cite: 65].

---

## 👨‍💻 Author
**[Ian Chen]** *M.S. Electrical and Computer Engineering* Focus: IC Design / Computer Architecture