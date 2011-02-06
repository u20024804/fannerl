-module(fannerl_tests).

-include_lib("eunit/include/eunit.hrl").

create_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    fannerl:get_training_algorithm(Fann).
    
get_mse_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    0.0 = fannerl:get_mse(Fann).

activation_function_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    fann_sigmoid_stepwise = fannerl:get_activation_function(Fann, 1, 1).
    
