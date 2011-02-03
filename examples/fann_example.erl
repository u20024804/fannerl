-module(fann_example).

-export([run/0, run2/0]).

run() ->
    Net = fann:create_standard({2,3,1}),
    fann:set_activation_function_hidden(Net, 5),
    fann:set_activation_function_output(Net, 5),
    fann:train_on_file(Net, "xor.data", 1000000, 0, 0.001),
    Mse = fann:get_mse(Net),
    Activation = fann:get_activation_function(Net, 1, 0),
    fann:save(Net, "xor.net"),
    {Mse, Activation, Net}.
   
run2() ->
    Net = fann:create_standard({2,3,1}),
    fann:set_activation_function_hidden(Net, 5),
    fann:set_activation_function_output(Net, 5),
    fann:train_on_data(Net, [[{-1,-1},{-1}],
			     [{-1,1}, {1}], 
			     [{1,-1},{1}], 
			     [{1,1}, {-1}]], 1000000, 0, 0.001),
    receive fann_train_on_data_complete -> ok end,
    Mse = fann:get_mse(Net),
    fann:set_activation_function_output(Net, 2),
    {-1.0}=fann:run(Net, {-1,-1}),
    {1.0}=fann:run(Net, {-1,1}),
    {1.0}=fann:run(Net, {1,-1}),
    {-1.0}=fann:run(Net, {1,1}),
    Mse.
