module gpu_core (
    input  wire        clk,
    input  wire        rst_n
    // ... External Memory Interface (Block RAM) ...
);

    // ============================================================
    // Internal Wire Declarations
    // ============================================================
    
    // Decoded control signals from the Control Unit
    wire        tensor_en;       // Enable signal for tensor operations
    wire        tensor_do_relu;  // Activation function (ReLU) trigger
    wire [4:0]  rs1_addr, rs2_addr, rs3_addr, rd_addr; // Register source/destination addresses
    wire        reg_we;          // Register write enable (internal)

    // Data outputs from Register File
    wire [63:0] reg_data_a;      // Multiplicand (A)
    wire [63:0] reg_data_b;      // Multiplier (B)
    wire [63:0] reg_data_c;      // Accumulator/Addend (C)

    // Outputs from Tensor Execution Unit
    wire [63:0] tensor_result;   // Final computed output
    wire        tensor_valid;    // High when result is valid (Ready for WB stage)
    
    // Pipeline registers for destination address tracking
    reg [4:0] rd_addr_q1, rd_addr_q2, rd_addr_delayed;

    // ============================================================
    // 1. Control Unit (Instruction Fetch & Decode)
    // ============================================================
    // Handles instruction sequencing and generates execution control signals
    gpu_control_unit u_control (
        .clk(clk),
        .rst_n(rst_n),
        // ... instruction fetch signals ...
        .out_tensor_en(tensor_en),
        .out_do_relu(tensor_do_relu),
        .out_rs1(rs1_addr),
        .out_rs2(rs2_addr),
        .out_rs3(rs3_addr),
        .out_rd(rd_addr)
        // ...
    );

    // ============================================================
    // 2. Register File (3R1W Architecture)
    // ============================================================
    // Supports 3-operand MAC operations (A*B+C) and 1 write-back port
    gpu_reg_file u_regfile (
        .clk(clk),
        .rst_n(rst_n),
        .read_addr_1(rs1_addr), .read_data_1(reg_data_a),
        .read_addr_2(rs2_addr), .read_data_2(reg_data_b),
        .read_addr_3(rs3_addr), .read_data_3(reg_data_c),
        
        // Write Control: Synchronized with pipeline completion
        .write_en(tensor_valid),       // Write-back only when operation is valid
        .write_addr(rd_addr_delayed),  // Uses the delayed destination address
        .write_data(tensor_result)
    );

    // ============================================================
    // 3. Tensor Processing Unit (TPU / Execution Stage)
    // ============================================================
    // Core compute logic for 64-bit vector/tensor operations
    tensor_unit_64bit u_tensor (
        .clk(clk),
        .rst_n(rst_n),
        .en(tensor_en),
        .do_relu(tensor_do_relu),
        .vector_a(reg_data_a),
        .vector_b(reg_data_b),
        .vector_c(reg_data_c),
        .vector_out(tensor_result),
        .valid_out(tensor_valid)
    );

    // ============================================================
    // Pipeline Synchronization Logic (Address Shift Register)
    // ============================================================
    // Since the Tensor Unit has a 3-cycle execution latency, 
    // the destination address (rd_addr) must be delayed by 3 cycles 
    // to align with the valid output data for the Write-Back stage.
    
    

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_addr_q1      <= 5'b0;
            rd_addr_q2      <= 5'b0;
            rd_addr_delayed <= 5'b0;
        end else begin
            rd_addr_q1      <= rd_addr;         // Latency Stage 1
            rd_addr_q2      <= rd_addr_q1;       // Latency Stage 2
            rd_addr_delayed <= rd_addr_q2;       // Latency Stage 3 (Synchronized with tensor_result)
        end
    end

endmodule