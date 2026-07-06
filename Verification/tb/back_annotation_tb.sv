`timescale 1ns / 100ps
import gemm_pkg::*;

module back_annotation_tb;

    // Memory arrays for testbench stimulus
    logic [ACT_WIDTH-1:0] activation_mem [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic [WGT_WIDTH-1:0] weight_mem     [0:ARRAY_SIZE-1][0:ARRAY_SIZE-1];
    logic [ACC_WIDTH-1:0] bias_mem       [0:ARRAY_SIZE-1];
    logic [ACC_WIDTH-1:0] scale_mem      [0:ARRAY_SIZE-1];
  

    // Clock and reset
    logic clk;
    logic rst_n;

    // Input Control signals
    logic TILE_START;
    logic TILE_LAST;
    logic BIAS_NEW;
    logic RELU_EN;
    logic [DIM_WIDTH-1:0] A_LEN;
    logic [DIM_WIDTH-1:0] W_LEN;
    logic [DIM_WIDTH-1:0] K_LEN;
    logic METADATA_VLD;
    logic DATA_VLD;
    logic BIAS_VLD;
    logic SCALE_VLD;
    logic Y_RDY;

    // Input Data signals
    logic [ACT_WIDTH-1:0] A_DATA [0:ARRAY_SIZE-1];
    logic [WGT_WIDTH-1:0] W_DATA [0:ARRAY_SIZE-1];
    logic [ACC_WIDTH-1:0] BIAS;
    logic [ACC_WIDTH-1:0] SCALE;

    // Output Control signals
    logic METADATA_RDY;
    logic DATA_RDY;
    logic BIAS_RDY;
    logic SCALE_RDY;
    logic [ARRAY_SIZE-1:0] Y_VLD; // add
    logic TILE_DONE;

    // Output Data signals
    logic [OUT_WIDTH-1:0] Y_OUT [0:ARRAY_SIZE-1];

    // Power / GND signals
    wire VPWR;
    wire VGND;
    
    assign VPWR = 1'b1;
    assign VGND = 1'b0;
    

    logic [31:0] A_DATA_bus;
    logic [31:0] W_DATA_bus;
    logic [31:0] Y_OUT_bus;

    assign A_DATA_bus = {
        A_DATA[7], A_DATA[6], A_DATA[5], A_DATA[4], A_DATA[3], A_DATA[2], A_DATA[1], A_DATA[0]};

    assign W_DATA_bus = {
        W_DATA[7], W_DATA[6], W_DATA[5], W_DATA[4], W_DATA[3], W_DATA[2], W_DATA[1], W_DATA[0]};

    assign {Y_OUT[7], Y_OUT[6], Y_OUT[5], Y_OUT[4], Y_OUT[3], Y_OUT[2], Y_OUT[1], Y_OUT[0]} = Y_OUT_bus;


    // Instantiate the GEMM Top Module
    gemm_top dut (
        .VPWR         (VPWR),
        .VGND         (VGND),
        .clk          (clk),
        .rst_n        (rst_n),
        .TILE_START   (TILE_START),
        .TILE_LAST    (TILE_LAST),
        .BIAS_NEW     (BIAS_NEW),
        .RELU_EN      (RELU_EN),
        .A_LEN        (A_LEN),
        .W_LEN        (W_LEN),
        .K_LEN        (K_LEN),
        .METADATA_VLD (METADATA_VLD),
        .DATA_VLD     (DATA_VLD),
        .BIAS_VLD     (BIAS_VLD),
        .SCALE_VLD    (SCALE_VLD),
        .Y_RDY        (Y_RDY),
        .A_DATA       (A_DATA_bus),
        .W_DATA       (W_DATA_bus),
        .BIAS         (BIAS),
        .SCALE        (SCALE),
        .METADATA_RDY (METADATA_RDY),
        .DATA_RDY     (DATA_RDY),
        .BIAS_RDY     (BIAS_RDY),
        .SCALE_RDY    (SCALE_RDY),
        .Y_VLD        (Y_VLD),
        .TILE_DONE    (TILE_DONE),
        .Y_OUT        (Y_OUT_bus)
    );

    // Clock generation
    initial clk = 1'b0;
    always #5 clk = ~clk;

    integer cycle;
    initial begin
        cycle = 0;
    end
    always @(posedge clk) begin
        cycle <= cycle + 1;
    end


     // Task to initialize inputs
    task automatic init_inputs();
        begin
            TILE_START   = 1'b0;
            TILE_LAST    = 1'b0;
            BIAS_NEW     = 1'b0;
            RELU_EN      = 1'b0;
            A_LEN        = '0;
            W_LEN        = '0;
            K_LEN        = '0;
            METADATA_VLD = 1'b0;
            DATA_VLD     = 1'b0;
            BIAS_VLD     = 1'b0;
            SCALE_VLD    = 1'b0;
            Y_RDY        = 1'b0;
            BIAS         = '0;
            SCALE        = '0;

            for (int i = 0; i < ARRAY_SIZE; i++) begin
                A_DATA[i] = '0;
                W_DATA[i] = '0;
            end
        end
    endtask

    // Task to send metadata to DUT at the beginning of each tile, during IDLE state
    task automatic send_metadata( input int a_dim, input int w_dim, input int k_dim, input logic tile_last_i, input logic bias_new_i, input logic relu_en_i);
        begin
            // Wait for DUT to be ready to receive metadata
            while (!METADATA_RDY) @(posedge clk);

            #5;
            // next clk, present metadata
            A_LEN        = a_dim;
            W_LEN        = w_dim;
            K_LEN        = k_dim;
            TILE_LAST    = tile_last_i;
            BIAS_NEW     = bias_new_i;
            RELU_EN      = relu_en_i;
            TILE_START   = 1'b1;
            METADATA_VLD = 1'b1;

            // next next clk, deassert start and metadata valid
            @(posedge clk);
            #5;
            TILE_START   = 1'b0;
            METADATA_VLD = 1'b0;
            A_LEN        = '0;
            W_LEN        = '0;
            K_LEN        = '0;
            TILE_LAST    = 1'b0;
            BIAS_NEW     = 1'b0;
            RELU_EN      = 1'b0;
        end
    endtask

    // Task to send tile data to FIFOs, 1 k-element vector per cycle for all rows/cols, during LOAD_FIFO state
    task automatic send_aw_tile(input int a_dim, input int w_dim, input int k_dim);
        begin

            while (!DATA_RDY) @(posedge clk); // Wait for DUT to assert ready before next beat
            #5;
            
            DATA_VLD <= 1'b1; // send data_vld every cycle while sending tile data

            for (int k = 0; k < k_dim; k++) begin // 1 clk send k-element vectors for all ARRAY_SIZE rows/cols
                for (int r = 0; r < ARRAY_SIZE; r++) begin // load activation column by column
                    A_DATA[r] = activation_mem[r][k]; 
                end
                for (int c = 0; c < ARRAY_SIZE; c++) begin // load weight row by row
                    W_DATA[c] = weight_mem[k][c];
                end

                @(posedge clk);
                #5;
            
            end

            DATA_VLD <= 1'b0;
            for (int i = 0; i < ARRAY_SIZE; i++) begin
                A_DATA[i] = '0;
                W_DATA[i] = '0;
            end
        end
    endtask

    // Task to send bias vector to PEs, 1 element per cycle during LOAD_FIFO state. 
    task automatic send_bias_vec(input int w_dim);
        begin
            
            while (!BIAS_RDY) @(posedge clk);
            #5;
            BIAS_VLD = 1'b1; // send bias_vld every cycle while sending bias vector

            for (int c = 0; c < w_dim; c++) begin // send a bias per clk for the first w_dim columns
                BIAS = bias_mem[c];
                @(posedge clk);
                #5; 
            end

            BIAS_VLD = 1'b0;
            BIAS     = '0;
        end
    endtask

    // Wait state = GEMM_COMPUTE, then wait a_dim+w_dim+k_dim-3 cycles 
    task automatic wait_gemm_compute(input int a_dim, input int w_dim, input int k_dim, input logic tile_last_i);
        begin
            int total_compute_cycles = a_dim + w_dim + k_dim - 1;
            // wait (state_output == GEMM_COMPUTE); // wait until we enter the compute state
            repeat (total_compute_cycles) @(posedge clk);
        end
    endtask

    // Task to drain output Y column by column, presenting corresponding scale for each column
    task automatic drain_outputs(input int w_dim);
        begin

            #5;
            for (int c = 0; c < w_dim; c++) begin
                SCALE_VLD = 1'b1; // tell DUT that the scale for this output column is valid
                Y_RDY     = 1'b1; // tell DUT that TB is ready to receive output data for this column
                SCALE = scale_mem[c];
                @(posedge clk); // send 1 scale per clk 
                #5;
            end


            @(posedge clk);
            #5;
            Y_RDY     = 1'b0;
            SCALE_VLD = 1'b0;
            SCALE     = '0;
        end
    endtask


    task automatic multi_tile_test(input int a_dim, input int w_dim, input int k_dim, input int num_tiles);
        begin
            //////////////
            // IDLE_STATE
            //////////////
            // Reset sequence
            rst_n = 1'b0;
            repeat (3) @(posedge clk);
            @(posedge clk);
            #5;
            rst_n = 1'b1;
            repeat (3) @(posedge clk);

            // send metadata for an 8x8 tile with bias and relu
            send_metadata(a_dim, w_dim, k_dim, 1'b0, 1'b1, 1'b1);

            //////////////////
            // LOAD_FIFO state
            //////////////////
            // send tile data and bias vector in parallel
            fork
                send_aw_tile(a_dim, w_dim, k_dim);
                send_bias_vec(w_dim);   
            join

            /////////////////////
            // GEMM_COMPUTE state
            /////////////////////
            // wait for compute to finish
            wait_gemm_compute(a_dim, w_dim, k_dim, 1'b0); // not the last tile

            for (int i = 0; i < num_tiles-2; i++) begin // repeat for num_tiles-2 times since we already did 1 tile above and 1 tile below
                ///////////////////////////
                // IDLE_STATE between tiles
                ///////////////////////////
                // send metadata for an 8x8 tile with bias and relu
                send_metadata(a_dim, w_dim, k_dim, 1'b0, 1'b0, 1'b1); // not the last tile and no new bias 

                //////////////////
                // LOAD_FIFO state for 2nd tile
                //////////////////
                // send tile data (same as 1st tile) and no bias this time in parallel
                send_aw_tile(a_dim, w_dim, k_dim);

                ////////////////
                // GEMM_COMPUTE state for 2nd tile
                ////////////////
                // wait for compute to finish
                wait_gemm_compute(a_dim, w_dim, k_dim, 1'b0); // not the last tile
            end
            ///////////////////////////
            // IDLE_STATE between tiles
            ///////////////////////////
            // send metadata for an 8x8 tile with bias and relu
            send_metadata(a_dim, w_dim, k_dim, 1'b1, 1'b0, 1'b1); // this is the last tile, but still no new bias

            ///////////////////////////////
            // LOAD_FIFO state for 3rd tile
            ///////////////////////////////
            // send tile data (same as 1st tile) and no bias this time in parallel
            send_aw_tile(a_dim, w_dim, k_dim);

            //////////////////////////////////
            // GEMM_COMPUTE state for 3rd tile
            //////////////////////////////////
            // wait for compute to finish
            wait_gemm_compute(a_dim, w_dim, k_dim, 1'b1); // this is the last tile

            ///////////////////
            // GEMM_FLUSH state
            ///////////////////
            // drain outputs and compare results
            drain_outputs(w_dim);
        end
    endtask

    // ----------------------------------------------------------------------
    // Test sequence
    // ----------------------------------------------------------------------

    always @(posedge clk) begin
        $display("[cycle %d] A=%h W=%h bias=%h scale=%h Y=%h metadata_rdy=%b bias_rdy=%b data_rdy=%b scale_rdy=%b y_vld=%1h tile_done=%b",
                  cycle, A_DATA_bus, W_DATA_bus, BIAS, SCALE, Y_OUT_bus,
                  METADATA_RDY, BIAS_RDY, DATA_RDY, SCALE_RDY, Y_VLD, TILE_DONE); 
    end

    initial begin
        // Load stimulus data from files
        $readmemh("activation_data.hex", activation_mem);
        $readmemh("weight_data.hex",     weight_mem);
        $readmemh("bias_data.hex",       bias_mem);
        $readmemh("scale_data.hex",      scale_mem);

        init_inputs();

        //---------------------
        // Multi-tile test
        //---------------------
        $display("========= Start Multi-tile with rectangular PE masking Test ! =========");
        multi_tile_test(3, 2, 2, 2);
        repeat(2) @(posedge clk);
        $display("========= Multi-tile with rectangular PE masking Test completed ! =========\n");


        $stop;
    end
endmodule

// vdel -all -lib work
// vlib work
// vmap work work

// vlog -vlog01compat -suppress 2892 +define+USE_POWER_PINS primitives.v
// vlog -vlog01compat -suppress 2892 +define+USE_POWER_PINS sky130_fd_sc_hd.v

// vlog -sv gemm_pkg.sv

// vlog -vlog01compat +define+USE_POWER_PINS gemm_pe.pnl.v
// vlog -vlog01compat +define+USE_POWER_PINS vector_unit.pnl.v
// vlog -vlog01compat +define+USE_POWER_PINS gemm_top.pnl.v

// vlog -sv tpu_back_annotation_tb_2.sv

// vsim -c work.tpu_back_annotation_tb_2 +sdf_verbose -sdfmax /tpu_back_annotation_tb_2/dut=gemm_top__nom_tt_025C_1v80.sdf -suppress 12088 -suppress 12090 -suppress 3262 -suppress 3438 -do "run -all; quit"

