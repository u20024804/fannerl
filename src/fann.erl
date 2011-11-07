%% fannerl (c) by Erik Axling

%% fannerl is licensed under a
%% Creative Commons Attribution-ShareAlike 3.0 Unported License.

%% You should have received a copy of the license along with this
%% work.  If not, see <http://creativecommons.org/licenses/by-sa/3.0/>.

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

