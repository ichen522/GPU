// ============================================================================
// Module 1: 17-bit Leading Zero Counter (Priority Encoder)
// ============================================================================
`timescale 1ns / 100ps
module lzc_17bit (
    input  wire [16:0] in_val,
    output reg  [4:0]  shift_amt,
    output reg         is_zero
);
    always @(*) begin
        is_zero = 1'b0;
        if      (in_val[16]) shift_amt = 5'd0;
        else if (in_val[15]) shift_amt = 5'd1;
        else if (in_val[14]) shift_amt = 5'd2;
        else if (in_val[13]) shift_amt = 5'd3;
        else if (in_val[12]) shift_amt = 5'd4;
        else if (in_val[11]) shift_amt = 5'd5;
        else if (in_val[10]) shift_amt = 5'd6;
        else if (in_val[9])  shift_amt = 5'd7;
        else if (in_val[8])  shift_amt = 5'd8;
        else if (in_val[7])  shift_amt = 5'd9;
        else if (in_val[6])  shift_amt = 5'd10;
        else if (in_val[5])  shift_amt = 5'd11;
        else if (in_val[4])  shift_amt = 5'd12;
        else if (in_val[3])  shift_amt = 5'd13;
        else if (in_val[2])  shift_amt = 5'd14;
        else if (in_val[1])  shift_amt = 5'd15;
        else if (in_val[0])  shift_amt = 5'd16;
        else begin
            shift_amt = 5'd17;
            is_zero = 1'b1;
        end
    end
endmodule

// ============================================================================
// Module 2: 3-Stage Pipelined BFloat16 MAC Core
// ============================================================================
module bf16_mac_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire [15:0] a,        // BFloat16 Input A
    input  wire [15:0] b,        // BFloat16 Input B
    input  wire [15:0] c,        // BFloat16 Accumulator/Partial Sum
    input  wire        do_relu,  // ReLU control
    output reg  [15:0] d,        // BFloat16 Result D
    output reg         valid_out
);

    // --- Stage 1: Multiply & Exponent Addition ---
    reg  [15:0] s1_c;
    reg         s1_sign_mul;
    reg  [8:0]  s1_exp_mul; 
    reg  [15:0] s1_mant_mul;
    reg         s1_valid;
    reg         s1_do_relu;

    wire [7:0] mant_a = {1'b1, a[6:0]}; // Append hidden bit
    wire [7:0] mant_b = {1'b1, b[6:0]};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid <= 1'b0;
        end else if (en) begin
            s1_sign_mul <= a[15] ^ b[15];
            s1_exp_mul  <= a[14:7] + b[14:7] - 8'd127;
            s1_mant_mul <= mant_a * mant_b;
            s1_c        <= c;
            s1_do_relu  <= do_relu;
            s1_valid    <= 1'b1;
        end else begin
            s1_valid <= 1'b0;
        end
    end

    // --- Stage 2: Alignment & Addition ---
    reg         s2_sign_res;
    reg  [8:0]  s2_exp_res;
    reg  [16:0] s2_mant_sum; 
    reg         s2_valid;
    reg         s2_do_relu;

    wire        c_sign = s1_c[15];
    wire [8:0]  c_exp  = {1'b0, s1_c[14:7]};
    wire [15:0] c_mant = {1'b1, s1_c[6:0], 8'b0}; // Scale to match 16-bit multiplier output

    reg [8:0]  exp_diff;
    reg [8:0]  larger_exp;
    reg [15:0] aligned_mul_mant;
    reg [15:0] aligned_c_mant;

    always @(*) begin
        if (s1_exp_mul > c_exp) begin
            exp_diff         = s1_exp_mul - c_exp;
            larger_exp       = s1_exp_mul;
            aligned_mul_mant = s1_mant_mul;
            aligned_c_mant   = c_mant >> exp_diff;
        end else begin
            exp_diff         = c_exp - s1_exp_mul;
            larger_exp       = c_exp;
            aligned_mul_mant = s1_mant_mul >> exp_diff;
            aligned_c_mant   = c_mant;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid <= 1'b0;
        end else begin
            s2_exp_res <= larger_exp;
            if (s1_sign_mul == c_sign) begin
                s2_mant_sum <= aligned_mul_mant + aligned_c_mant;
                s2_sign_res <= s1_sign_mul;
            end else begin
                if (aligned_mul_mant > aligned_c_mant) begin
                    s2_mant_sum <= aligned_mul_mant - aligned_c_mant;
                    s2_sign_res <= s1_sign_mul;
                end else begin
                    s2_mant_sum <= aligned_c_mant - aligned_mul_mant;
                    s2_sign_res <= c_sign;
                end
            end
            s2_do_relu <= s1_do_relu;
            s2_valid   <= s1_valid;
        end
    end

    // --- Stage 3: Normalization & ReLU ---
    wire [4:0] shift_amount;
    wire       mant_is_zero;

    lzc_17bit u_lzc (
        .in_val(s2_mant_sum),
        .shift_amt(shift_amount),
        .is_zero(mant_is_zero)
    );

    wire [16:0] norm_mant = s2_mant_sum << shift_amount;
    wire [8:0]  norm_exp  = s2_exp_res - {4'b0, shift_amount};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            d         <= 16'b0;
        end else begin
            if (s2_valid) begin
                valid_out <= 1'b1;
                if (mant_is_zero) begin
                    d <= 16'b0;
                end else begin
                    d[15]   <= s2_sign_res;
                    d[14:7] <= norm_exp[7:0];
                    d[6:0]  <= norm_mant[15:9]; // Truncate to 7-bit mantissa
                end
                
                // ReLU Activation Function
                if (s2_do_relu && s2_sign_res) begin
                    d <= 16'b0;
                end
            end else begin
                valid_out <= 1'b0;
            end
        end
    end
endmodule

// ============================================================================
// Module 3: Top-Level 64-bit Tensor Unit (SIMD wrapper)
// ============================================================================
module tensor_unit_64bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire [63:0] vector_a,
    input  wire [63:0] vector_b,
    input  wire [63:0] vector_c,
    input  wire        do_relu,
    output wire [63:0] vector_out,
    output wire        valid_out
);

    wire v0, v1, v2, v3;
    
    // valid_out is high only when all lanes complete (they operate in lockstep)
    assign valid_out = v0 & v1 & v2 & v3;

    // Instantiate 4 MAC lanes for SIMD execution
    bf16_mac_core lane0 (
        .clk(clk), .rst_n(rst_n), .en(en), .do_relu(do_relu),
        .a(vector_a[15:0]), .b(vector_b[15:0]), .c(vector_c[15:0]),
        .d(vector_out[15:0]), .valid_out(v0)
    );

    bf16_mac_core lane1 (
        .clk(clk), .rst_n(rst_n), .en(en), .do_relu(do_relu),
        .a(vector_a[31:16]), .b(vector_b[31:16]), .c(vector_c[31:16]),
        .d(vector_out[31:16]), .valid_out(v1)
    );

    bf16_mac_core lane2 (
        .clk(clk), .rst_n(rst_n), .en(en), .do_relu(do_relu),
        .a(vector_a[47:32]), .b(vector_b[47:32]), .c(vector_c[47:32]),
        .d(vector_out[47:32]), .valid_out(v2)
    );

    bf16_mac_core lane3 (
        .clk(clk), .rst_n(rst_n), .en(en), .do_relu(do_relu),
        .a(vector_a[63:48]), .b(vector_b[63:48]), .c(vector_c[63:48]),
        .d(vector_out[63:48]), .valid_out(v3)
    );

endmodule