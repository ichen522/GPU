`timescale 1ns / 100ps

module tb_tensor_unit_64bit();

    // ============================================================
    // 1. Testbench Signal Declarations
    // ============================================================
    reg         clk;
    reg         rst_n;
    reg         en;
    reg  [63:0] vector_a;
    reg  [63:0] vector_b;
    reg  [63:0] vector_c;
    reg         do_relu;

    wire [63:0] vector_out;
    wire        valid_out;

    // ============================================================
    // 2. Device Under Test (DUT) Instantiation
    // ============================================================
    tensor_unit_64bit dut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .vector_a(vector_a),
        .vector_b(vector_b),
        .vector_c(vector_c),
        .do_relu(do_relu),
        .vector_out(vector_out),
        .valid_out(valid_out)
    );

    // ============================================================
    // 3. Clock Generation
    // ============================================================
    // 10ns clock period (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // ============================================================
    // 4. Test Stimulus
    // ============================================================
    initial begin
        // --- Signal Initialization ---
        rst_n    = 0;
        en       = 0;
        do_relu  = 0;
        vector_a = 64'b0;
        vector_b = 64'b0;
        vector_c = 64'b0;

        // --- System Reset ---
        #15;
        rst_n = 1;
        #10;

        $display("==================================================");
        $display("   Starting BFloat16 Tensor Unit Simulation");
        $display("==================================================");

        // -----------------------------------------------------------
        // Test Case 1: Standard MAC Operation (ReLU Disabled)
        // -----------------------------------------------------------
        // Data layout: 4 BFloat16 lanes across 64 bits.
        // Lane 0: A=2.0 (16'h4000), B=3.0 (16'h4040), C=1.0 (16'h3f80) -> Expected D=7.0 (16'h40e0)
        // Lane 1: A=1.0 (16'h3f80), B=1.0 (16'h3f80), C=0.0 (16'h0000) -> Expected D=1.0 (16'h3f80)
        // Lane 2: A=-2.0(16'hc000), B=2.0 (16'h4000), C=1.0 (16'h3f80) -> Expected D=-3.0(16'hc040)
        // Lane 3: A=0.0 (16'h0000), B=0.0 (16'h0000), C=2.0 (16'h4000) -> Expected D=2.0 (16'h4000)
        
        @(posedge clk);
        en       = 1;
        do_relu  = 0;
        vector_a = {16'h0000, 16'hc000, 16'h3f80, 16'h4000};
        vector_b = {16'h0000, 16'h4000, 16'h3f80, 16'h4040};
        vector_c = {16'h4000, 16'h3f80, 16'h0000, 16'h3f80};

        // De-assert enable to simulate a single instruction issue
        @(posedge clk);
        en = 0; 

        // Wait for pipeline to flush (monitor valid_out)
        wait(valid_out == 1'b1);
        $display("[Test 1: MAC w/o ReLU] Done");
        $display("Expected Output: 4000_c040_3f80_40e0");
        $display("Actual Output:   %x_%x_%x_%x", 
                 vector_out[63:48], vector_out[47:32], vector_out[31:16], vector_out[15:0]);
        
        // -----------------------------------------------------------
        // Test Case 2: MAC Operation + ReLU Enabled
        // -----------------------------------------------------------
        // Testing if Lane 2's negative result (-3.0) is clipped to 0.0
        
        @(posedge clk);
        en       = 1;
        do_relu  = 1; // Enable ReLU activation
        vector_a = {16'h0000, 16'hc000, 16'h3f80, 16'h4000};
        vector_b = {16'h0000, 16'h4000, 16'h3f80, 16'h4040};
        vector_c = {16'h4000, 16'h3f80, 16'h0000, 16'h3f80};

        @(posedge clk);
        en = 0; 

        wait(valid_out == 1'b1);
        $display("\n[Test 2: MAC w/ ReLU] Done");
        $display("Expected Output: 4000_0000_3f80_40e0  <-- Lane 2 clipped to zero");
        $display("Actual Output:   %x_%x_%x_%x", 
                 vector_out[63:48], vector_out[47:32], vector_out[31:16], vector_out[15:0]);

        $display("==================================================");
        $display("   Simulation Completed");
        $display("==================================================");
        #20 $finish; 
    end

endmodule