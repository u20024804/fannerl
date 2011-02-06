-module(fannerl).

-export([create_standard/1, train_on_file/5, get_mse/1, save/2,
	 set_activation_function_hidden/2, set_activation_function_output/2,
	 get_activation_function/3, print_parameters/1, print_connections/1,
	 run/2, test/3, randomize_weights/3, train_on_data/5, create_train/1,
	 shuffle_train_data/1, scale_train/2, descale_train/2, 
	 set_input_scaling_params/4, set_output_scaling_params/4,
	 set_scaling_params/6, clear_scaling_params/1, scale_input_train_data/3,
	 scale_output_train_data/3, scale_train_data/3, merge_train_data/2,
	 subset_train_data/3, num_input_train_data/1, num_output_train_data/1,
	 save_train/2, get_training_algorithm/1, set_training_algorithm/2,
	 get_learning_rate/1, set_learning_rate/2, get_learning_momentum/1,
	 set_learning_momentum/2, length_train_data/1,  
	 set_activation_function_layer/3, get_activation_steepness/3,
	 set_activation_steepness/4, set_activation_steepness_layer/3,
	 set_activation_steepness_hidden/2, set_activation_steepness_output/2,
	 get_train_error_function/1, set_train_error_function/2,
	 get_train_stop_function/1, set_train_stop_function/2,
	 get_bit_fail_limit/1, set_bit_fail_limit/2,	 
	 get_quickprop_mu/1, set_quickprop_mu/2, 
	 get_rprop_increase_factor/1, set_rprop_increase_factor/2,
	 get_rprop_decrease_factor/1, set_rprop_decrease_factor/2,
	 get_rprop_delta_min/1, set_rprop_delta_min/2,
	 get_rprop_delta_max/1, set_rprop_delta_max/2,
	 get_rprop_delta_zero/1, set_rprop_delta_zero/2,
	 get_bit_fail/1, reset_mse/1, train_epoch/2, test_data/2, 
	 set_activation_function/4]).

-on_load(init/0).

init() ->
    case code:where_is_file("fannerl.beam") of
	non_existing ->
	    fannerl_nif_library_not_loaded;
	AbsPath ->
	    erlang:load_nif(filename:join(filename:dirname(AbsPath), 
					  "../priv/fannerl_nif"), 0)
    end.

create_standard(_) ->
    exit(nif_library_not_loaded).

train_on_file(_,_,_,_,_) ->
    exit(nif_library_not_loaded).

get_mse(_) ->
    exit(nif_library_not_loaded).

save(_,_) ->
    exit(nif_library_not_loaded).

set_activation_function_hidden(_,_) ->
    exit(nif_library_not_loaded).

set_activation_function_output(_,_) ->
    exit(nif_library_not_loaded).

set_activation_function(_,_,_,_) ->
    exit(nif_library_not_loaded).

get_activation_function(_,_,_) ->
    exit(nif_library_not_loaded).

print_parameters(_) ->
    exit(nif_library_not_loaded).

print_connections(_) ->
    exit(nif_library_not_loaded).

run(_,_) ->
    exit(nif_library_not_loaded).

test(_,_,_) ->
    exit(nif_library_not_loaded).

randomize_weights(_,_,_) ->
    exit(nif_library_not_loaded).

train_on_data(_,_,_,_,_) ->
    exit(nif_library_not_loaded).

create_train(_) ->
    exit(nif_library_not_loaded).

shuffle_train_data(_) ->
    exit(nif_library_not_loaded).
 
scale_train(_,_) ->
    exit(nif_library_not_loaded).
descale_train(_,_) ->

    exit(nif_library_not_loaded).
set_input_scaling_params(_,_,_,_) ->
    exit(nif_library_not_loaded).

set_output_scaling_params(_,_,_,_) ->
    exit(nif_library_not_loaded).

set_scaling_params(_,_,_,_,_,_) ->
    exit(nif_library_not_loaded).

clear_scaling_params(_) ->
    exit(nif_library_not_loaded).

scale_input_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

scale_output_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

scale_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

merge_train_data(_,_) ->
    exit(nif_library_not_loaded).

subset_train_data(_,_,_) ->
    exit(nif_library_not_loaded).

num_input_train_data(_) ->
    exit(nif_library_not_loaded).

num_output_train_data(_) ->
    exit(nif_library_not_loaded).

save_train(_,_) ->
    exit(nif_library_not_loaded).

get_training_algorithm(_) ->
    exit(nif_library_not_loaded).

set_training_algorithm(_,_) ->
    exit(nif_library_not_loaded).

get_learning_rate(_) ->
    exit(nif_library_not_loaded).

set_learning_rate(_,_) ->
    exit(nif_library_not_loaded).

get_learning_momentum(_) ->
    exit(nif_library_not_loaded).

set_learning_momentum(_,_) ->
    exit(nif_library_not_loaded).

length_train_data(_) ->
    exit(nif_library_not_loaded).

set_activation_function_layer(_,_,_) ->
    exit(nif_library_not_loaded).

get_activation_steepness(_,_,_) ->
    exit(nif_library_not_loaded).

set_activation_steepness(_,_,_,_) ->
    exit(nif_library_not_loaded).

set_activation_steepness_layer(_,_,_) ->
    exit(nif_library_not_loaded).

set_activation_steepness_hidden(_,_) ->
    exit(nif_library_not_loaded).

set_activation_steepness_output(_,_) ->
    exit(nif_library_not_loaded).
    
get_train_error_function(_) ->
    exit(nif_library_not_loaded).

set_train_error_function(_,_) ->
    exit(nif_library_not_loaded).

get_train_stop_function(_) ->
    exit(nif_library_not_loaded).

set_train_stop_function(_,_) ->
    exit(nif_library_not_loaded).

get_bit_fail_limit(_) ->
    exit(nif_library_not_loaded).

set_bit_fail_limit(_,_) ->
    exit(nif_library_not_loaded).

get_quickprop_mu(_) ->
    exit(nif_library_not_loaded). 

set_quickprop_mu(_,_) ->
    exit(nif_library_not_loaded).

get_rprop_increase_factor(_) ->
    exit(nif_library_not_loaded).

set_rprop_increase_factor(_,_) ->
    exit(nif_library_not_loaded).

get_rprop_decrease_factor(_) ->
    exit(nif_library_not_loaded).

set_rprop_decrease_factor(_,_) ->
    exit(nif_library_not_loaded).

get_rprop_delta_min(_) ->
    exit(nif_library_not_loaded).

set_rprop_delta_min(_,_) ->
    exit(nif_library_not_loaded).

get_rprop_delta_max(_) ->
    exit(nif_library_not_loaded).

set_rprop_delta_max(_,_) ->
    exit(nif_library_not_loaded).

get_rprop_delta_zero(_) ->
    exit(nif_library_not_loaded).

set_rprop_delta_zero(_,_) ->
    exit(nif_library_not_loaded).

get_bit_fail(_) ->
    exit(nif_library_not_loaded). 

reset_mse(_) ->
    exit(nif_library_not_loaded).

train_epoch(_,_) ->
    exit(nif_library_not_loaded).

test_data(_,_) ->
    exit(nif_library_not_loaded).
