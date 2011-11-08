-module(fannerl_tests).

-include_lib("eunit/include/eunit.hrl").

create_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    fannerl:get_training_algorithm(Fann),
    ?debugMsg("exiting create_test").
    
get_mse_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    0.0 = fannerl:get_mse(Fann),
    ?debugMsg("exiting get_mse_test"),
    erlang:garbage_collect(),
    timer:sleep(1000).

activation_function_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    fann_sigmoid_stepwise = fannerl:get_activation_function(Fann, 1, 1),
    ok = fannerl:set_activation_function(Fann, fann_elliot, 1, 1),
    fann_elliot = fannerl:get_activation_function(Fann, 1, 1),
    ok = fannerl:set_activation_function(Fann, fann_sigmoid, 1, 1),
    fann_sigmoid = fannerl:get_activation_function(Fann, 1, 1).

    

failing_segment_fault_test() ->
    Fann = fannerl:create_standard({2,2,1}),
    fann_sigmoid_stepwise = fannerl:get_activation_function(Fann, 1, 1),
    ok = fannerl:set_activation_function(Fann, fann_sigmoid, 1, 1),
    fann_sigmoid = fannerl:get_activation_function(Fann, 1, 1).
