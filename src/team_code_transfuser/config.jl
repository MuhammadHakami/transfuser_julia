module configs
    Base.@kwdef struct GlobalConfig
        root_dir=""
        setting="all"
        train_towns::Vector{String} = []
        val_towns::Vector{String} = []
        train_data::Vector{String} = []
        val_data::Vector{String} = []
        """ base architecture configurations """
        # Data
        seq_len::Int64 = 1 # input timesteps
        # use different seq len for image and lidar
        img_seq_len::Int64 = 1 
        lidar_seq_len::Int64 = 1
        pred_len::Int64 = 4 # future waypoints predicted
        scale::Int64 = 1 # image pre-processing
        img_resolution::Tuple{Int64, Int64} = (160, 704) # image pre-processing in H, W
        img_width::Int64 = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
        lidar_resolution_width::Int64  = 256 # Width of the LiDAR grid that the point cloud is voxelized into.
        lidar_resolution_height::Int64 = 256 # Height of the LiDAR grid that the point cloud is voxelized into.
        pixels_per_meter::Int64 = 8.0 # How many pixels make up 1 meter. 1 / pixels_per_meter = size of pixel in meters
        lidar_pos::Vector{Float64} = [1.3,0.0,2.5] # x, y, z mounting position of the LiDAR
        lidar_rot::Vector{Float64} = [0.0, 0.0, -90.0] # Roll Pitch Yaw of LiDAR in degree

        camera_pos::Vector{Float64} = [1.3, 0.0, 2.3] #x, y, z mounting position of the camera
        camera_width::Int64 = 960 # Camera width in pixel
        camera_height::Int64 = 480 # Camera height in pixel
        camera_fov::Int64 = 120 #Camera FOV in degree
        camera_rot_0::Vector{Float64} = [0.0, 0.0, 0.0] # Roll Pitch Yaw of camera 0 in degree
        camera_rot_1::Vector{Float64} = [0.0, 0.0, -60.0] # Roll Pitch Yaw of camera 1 in degree
        camera_rot_2::Vector{Float64} = [0.0, 0.0, 60.0] # Roll Pitch Yaw of camera 2 in degree

        bev_resolution_width::Int16  = 160 # Width resoultion the BEV loss is upsampled to. Double check if width and height are swapped if you want to make them non symmetric.
        bev_resolution_height::Int16 = 160 # Height resoultion the BEV loss is upsampled to. Double check if width and height are swapped if you want to make them non symmetric.
        use_target_point_image::Bool = false
        gru_concat_target_point::Bool = true
        augment::Bool = true
        inv_augment_prob::Float64 = 0.1 # Probablity that data augmentation is applied is 1.0 - inv_augment_prob
        aug_max_rotation::Int64 = 20 # degree
        debug::Bool = false # If true the model in and outputs will be visualized and saved into Os variable Save_Path
        sync_batch_norm::Bool = false # If this is true we convert the batch norms, to synced bach norms.
        train_debug_save_freq::Int64 = 50 # At which interval to save debug files to disk during training

        bb_confidence_threshold::Float64 = 0.3 # Confidence of a bounding box that is needed for the detection to be accepted

        # Lidar discretization, configuration only used for Point Pillars
        use_point_pillars::Bool = false
        max_lidar_points::Int64 = 40000
        min_x::Int64 = -16
        max_x::Int64 = 16
        min_y::Int64 = -32
        max_y::Int64 = 0
        num_input::Int64 = 9
        num_features::Vector{Int64} = [32, 32]

        backbone = "transFuser"

        # CenterNet parameters
        num_dir_bins = 12
        fp16_enabled = false
        center_net_bias_init_with_prob = 0.1
        center_net_normal_init_std = 0.001
        top_k_center_keypoints = 100
        center_net_max_pooling_kernel = 3
        channel = 64

        bounding_box_divisor = 2.0 # The height and width of the bounding box value was changed by this factor during data collection. Fix that for future datasets and remove
        draw_brake_threshhold = 0.5 # If the brake value is higher than this threshhold, the bb will be drawn with the brake color during visualization

        #Waypoint GRU
        gru_hidden_size = 64

        num_class = 7
        classes = Dict(
            0=> [0, 0, 0],  # unlabeled
            1=> [0, 0, 255],  # vehicle
            2=> [128, 64, 128],  # road
            3=> [255, 0, 0],  # red light
            4=> [0, 255, 0],  # pedestrian
            5=> [157, 234, 50],  # road line
            6=> [255, 255, 255],  # sidewalk
        )
        #Color format BGR
        classes_list = [
            [0, 0, 0],  # unlabeled
            [255, 0, 0],  # vehicle
            [128, 64, 128],  # road
            [0, 0, 255],  # red light
            [0, 255, 0],  # pedestrian
            [50, 234, 157],  # road line
            [255, 255, 255],  # sidewalk
        ]
        converter = [
            0,  # unlabeled
            0,  # building
            0,  # fence
            0,  # other
            4,  # pedestrian
            0,  # pole
            5,  # road line
            2,  # road
            6,  # sidewalk
            0,  # vegetation
            1,  # vehicle
            0,  # wall
            0,  # traffic sign
            0,  # sky
            0,  # ground
            0,  # bridge
            0,  # rail track
            0,  # guard rail
            0,  # traffic light
            0,  # static
            0,  # dynamic
            0,  # water
            0,  # terrain
            3,  # red light
            3,  # yellow light
            0,  # green light
            0,  # stop sign
            5,  # stop line marking
        ]

        # Optimization
        lr = 1e-4 # learning rate
        multitask = true # whether to use segmentation + depth losses
        ls_seg   = 1.0
        ls_depth = 10.0

        # Conv Encoder
        img_vert_anchors = 5
        img_horz_anchors = 20 + 2
        lidar_vert_anchors = 8
        lidar_horz_anchors = 8
        
        img_anchors = img_vert_anchors * img_horz_anchors
        lidar_anchors = lidar_vert_anchors * lidar_horz_anchors

        detailed_losses = ["loss_wp", "loss_bev", "loss_depth", "loss_semantic", "loss_center_heatmap", "loss_wh",
                        "loss_offset", "loss_yaw_class", "loss_yaw_res", "loss_velocity", "loss_brake"]
        detailed_losses_weights = [1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

        perception_output_features = 512 # Number of features outputted by the perception branch.
        bev_features_chanels = 64 # Number of channels for the BEV feature pyramid
        bev_upsample_factor = 2

        deconv_channel_num_1 = 128 # Number of channels at the first deconvolution layer
        deconv_channel_num_2 = 64 # Number of channels at the second deconvolution layer
        deconv_channel_num_3 = 32 # Number of channels at the third deconvolution layer

        deconv_scale_factor_1 = 8 # Scale factor, of how much the grid size will be interpolated after the first layer
        deconv_scale_factor_2 = 4 # Scale factor, of how much the grid size will be interpolated after the second layer

        gps_buffer_max_len = 100 # Number of past gps measurements that we track.
        carla_frame_rate = 1.0 / 20.0 # CARLA frame rate in milliseconds
        carla_fps = 20 # Simulator Frames per second
        iou_treshold_nms = 0.2  # Iou threshold used for Non Maximum suppression on the Bounding Box predictions for the ensembles
        steer_damping = 0.5 # Damping factor by which the steering will be multiplied when braking
        route_planner_min_distance = 7.5
        route_planner_max_distance = 50.0
        action_repeat = 2 # Number of times we repeat the networks action. It"s 2 because the LiDAR operates at half the frame rate of the simulation
        stuck_threshold = 1100/action_repeat # Number of frames after which the creep controller starts triggering. Divided by
        creep_duration = 30 / action_repeat # Number of frames we will creep forward

        # Size of the safety box
        safety_box_z_min = -2.0
        safety_box_z_max = -1.05

        safety_box_y_min = -3.0
        safety_box_y_max = 0.0

        safety_box_x_min = -1.066
        safety_box_x_max = 1.066

        ego_extent_x = 2.4508416652679443 # Half the length of the ego car in x direction
        ego_extent_y = 1.0641621351242065 # Half the length of the ego car in x direction
        ego_extent_z = 0.7553732395172119 # Half the length of the ego car in x direction

        # GPT Encoder
        n_embd = 512
        block_exp = 4
        n_layer = 8
        n_head = 4
        n_scale = 4
        embd_pdrop = 0.1
        resid_pdrop = 0.1
        attn_pdrop = 0.1
        gpt_linear_layer_init_mean = 0.0 # Mean of the normal distribution with which the linear layers in the GPT are initialized
        gpt_linear_layer_init_std  = 0.02 # Std  of the normal distribution with which the linear layers in the GPT are initialized
        gpt_layer_norm_init_weight = 1.0 # Initial weight of the layer norms in the gpt.

        # Controller
        turn_KP = 1.25
        turn_KI = 0.75
        turn_KD = 0.3
        turn_n = 20 # buffer size

        speed_KP = 5.0
        speed_KI = 0.5
        speed_KD = 1.0
        speed_n = 20 # buffer size

        default_speed = 4.0 # Speed used when creeping

        max_throttle = 0.75 # upper limit on throttle signal value in dataset
        brake_speed = 0.4 # desired speed below which brake is triggered
        brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
        clip_delta = 0.25 # maximum change in speed input to logitudinal controller
        clip_throttle = 0.75 # Maximum throttle allowed by the controller

        function GlobalConfig(root_dir="", setting="all", train_towns=[], val_towns=[], train_data=[], val_data=[], kwargs...)
            # println(opt)
            # root_dir = root_dir
            if setting =="all"
                train_towns = readdir(root_dir)
                val_towns = [train_towns[1]]
                for town in train_towns
                    root_files = readdir(joinpath(root_dir, town))
                    for file in root_files
                        if ! isfile(joinpath(root_dir, file))
                            push!(train_data, joinpath(root_dir, town, file))
                        end
                    end
                end
                for town in val_towns
                    root_files = readdir(joinpath(root_dir, town))
                    for file in root_files
                        if ! isfile(joinpath(root_dir, file))
                            push!(val_data, joinpath(root_dir, town, file))
                        end
                    end
                end
            elseif setting=="02_05_withheld"
                println("Skip Town02 and Town05")
                train_towns = readdir(root_dir)
                val_towns = train_towns
                for town in train_towns
                    root_files = readdir(joinpath(root_dir, town))
                    for file in root_files
                        if (typeof(findfirst("Town02", file))!=Nothing) || (typeof(findfirst("Town05", file))==Nothing)
                            continue
                        end
                        if ! isfile(joinpath(root_dir, file))
                            print("Train Folder: ", file)
                            push!(train_data, joinpath(root_dir, town, file))
                        end
                    end
                end
                for town in val_towns
                    root_files = readdir(joinpath(root_dir, town))
                    for file in root_files
                        if (typeof(findfirst("Town02", file))==Nothing) && (typeof(findfirst("Town05", file))==Nothing)
                            continue
                        end
                        if ! isfile(joinpath(root_dir, file))
                            println("val Folder: ", file)
                            push!(val_data, joinpath(root_dir, town, file))
                        end
                    end
                end
            elseif (setting =="eval")
                Nothing
            else
                println("Error: Selected Setting: ", setting, " does not exist.")
            end
            # println("",kwargs...)
            new(root_dir, setting, train_towns, val_towns, train_data, val_data, kwargs...)
        end
        # GlobalConfig() = GlobalConfigA(new())
    end
end
export GlobalConfig

# configs = GlobalConfig(root_dir="/media/mokami/autonomous_vehicles/repos/transfuser/data", setting="all", clip_throttle=0.7)

# train_data = ["/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town01_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town02_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town03_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town04_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town05_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town06_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town07_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town10HD_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town01_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town02_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town03_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town04_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town05_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town06_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town07_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/cycl_dataset_23_11/Routes_non-straight-junction-Scen4_Town10HD_junction_Seed2000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town01_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town02_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town03_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town04_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town05_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town06_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town07_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/dirt_dataset_23_11/Routes_Scenario1_Town10HD_curved_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town01_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town02_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town03_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town04_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town05_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town06_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town07_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_l_dataset_23_11/Routes_routes_30mshortroutes_Town10HD_Scenario8junction_Seed4000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town01_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town02_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town03_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town04_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town05_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town06_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town07_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_r_dataset_23_11/Routes_routes_30mshortroutes_Town10HD_Scenario9_Seed5000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town01_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town02_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town03_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town04_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town05_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town06_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town07_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_s_dataset_23_11/Routes_routes_30mshortroutes_Town10HD_Scenario7junction_Seed3000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_u_dataset_23_11/Routes_Scenenario10_routes_Town03_junction_Seed6000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_u_dataset_23_11/Routes_Scenenario10_routes_Town04_junction_Seed6000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_u_dataset_23_11/Routes_Scenenario10_routes_Town05_junction_Seed6000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_u_dataset_23_11/Routes_Scenenario10_routes_Town06_junction_Seed6000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_u_dataset_23_11/Routes_Scenenario10_routes_Town07_junction_Seed6000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/int_u_dataset_23_11/Routes_Scenenario10_routes_Town10HD_junction_Seed6000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town01_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town02_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town03_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town04_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town05_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town06_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town07_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/left_dataset_23_11/Routes_routes_10mshortroutes_Town10HD_Scenario9_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/ll_dataset_23_11/Routes_clipped_Town04_ll_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/ll_dataset_23_11/Routes_clipped_Town05_ll_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/ll_dataset_23_11/Routes_clipped_Town06_ll_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/lr_dataset_23_11/Routes_clipped_Town03_lr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/lr_dataset_23_11/Routes_clipped_Town04_lr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/lr_dataset_23_11/Routes_clipped_Town05_lr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/lr_dataset_23_11/Routes_clipped_Town06_lr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/lr_dataset_23_11/Routes_clipped_Town10HD_lr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town01_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town02_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town03_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town04_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town05_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town06_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town07_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/right_dataset_23_11/Routes_routes_10mshortroutes_Town10HD_Scenario8junction_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rl_dataset_23_11/Routes_clipped_Town03_rl_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rl_dataset_23_11/Routes_clipped_Town04_rl_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rl_dataset_23_11/Routes_clipped_Town05_rl_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rl_dataset_23_11/Routes_clipped_Town06_rl_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rl_dataset_23_11/Routes_clipped_Town10HD_rl_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rr_dataset_23_11/Routes_clipped_Town04_rr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rr_dataset_23_11/Routes_clipped_Town05_rr_Seed0", "/media/mokami/autonomous_vehicles/repos/transfuser/data/rr_dataset_23_11/Routes_clipped_Town06_rr_Seed0"]
# val_data = ["/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town01_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town02_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town03_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town04_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town05_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town06_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town07_curved_Seed1000", "/media/mokami/autonomous_vehicles/repos/transfuser/data/coke_dataset_23_11/Routes_Scenario3_Town10HD_curved_Seed1000"]
# @assert sort(configs.train_data)==sort(train_data)
# @assert sort(configs.train_data)==sort(train_data)
