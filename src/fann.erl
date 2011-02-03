-module(fann).

-behaviour(application).

-export([start/2, stop/1]).

%% ===================================================================
%% Application callbacks
%% ===================================================================

start(_StartType, _StartArgs) ->
    ok.

stop(_State) ->
    ok.

