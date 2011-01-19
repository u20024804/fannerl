#include <erl_nif.h>
#include <fann.h>

static ErlNifResourceType * FANN_POINTER=NULL;

int check_and_convert_array(ErlNifEnv* env, const ERL_NIF_TERM ** tuple_array,
			    int tuple_size, unsigned int * converted_array) {
  int i;
  unsigned int array_value;
  ERL_NIF_TERM term;
  int * point;
  for(i = 0; i < tuple_size; ++i) {
    term = *(*tuple_array + i);
    if(enif_get_uint(env, term, &array_value)) {
      point = converted_array + i;
      *point =  array_value;
    } else {
      return 0;
    }
  }
  return 1;
}

static int load(ErlNifEnv * env, void ** priv_data, ERL_NIF_TERM load_info){
  FANN_POINTER = enif_open_resource_type(env, 
					 NULL, 
					 "fann_pointer", 
					 NULL, 
					 ERL_NIF_RT_CREATE |
					 ERL_NIF_RT_TAKEOVER,
					 NULL);
  if(FANN_POINTER == NULL) {
    return -1;
  } else {
    return 0;
  }
}
  

static ERL_NIF_TERM create_standard_nif(ErlNifEnv* env, int argc,
					const ERL_NIF_TERM argv[]) {
  int x, ret, tuple_size;
  const ERL_NIF_TERM ** tuple_array;
  unsigned int * converted_array;
  struct fann * ann; 
  ERL_NIF_TERM result;
  ann = enif_alloc_resource(FANN_POINTER, sizeof(struct fann *));
  if(enif_get_tuple(env, argv[0], &tuple_size, tuple_array)) {
    converted_array = malloc(tuple_size*sizeof(unsigned int));
    if(check_and_convert_array(env, tuple_array, tuple_size, converted_array)) {
      ann = fann_create_standard_array(tuple_size, 
				       converted_array);
      if(converted_array!=NULL) {
	free(converted_array);
	converted_array=NULL;
      }      
      result = enif_make_resource(env, (void *)ann);
      enif_release_resource(ann);
      return result;
    } else {
      if(converted_array!=NULL) {
	free(converted_array);
	converted_array=NULL;
      }
      enif_release_resource(ann);
      return enif_make_badarg(env);
    }
  } else {
    enif_release_resource(ann);
    return enif_make_badarg(env);
  }
}

static ERL_NIF_TERM train_on_file_nif(ErlNifEnv* env, int argc,
				      const ERL_NIF_TERM argv[]) {
  struct fann * ann;
  unsigned int string_length, max_epochs, epochs_between_reports;
  char * file_name;
  double desired_error;
  
  printf("Before resource\n");
  if(!enif_get_resource(env, argv[0], FANN_POINTER, (void **)&ann)) {
    return enif_make_badarg(env);
  }
  printf("Before get length\n");
  if(!enif_get_list_length(env, argv[1], &string_length)){
    return enif_make_badarg(env);
  }
  printf("Before get_string\n");
  file_name = malloc(string_length*sizeof(char));
  enif_get_string(env, argv[1], file_name, string_length, ERL_NIF_LATIN1);
  printf("Before get first uint\n");
  if(!enif_get_uint(env, argv[2], &max_epochs)) {
    return enif_make_badarg(env);
  }
  printf("Before get second uint\n");
  if(!enif_get_uint(env, argv[3], &epochs_between_reports)) {
    return enif_make_badarg(env);
  }
  printf("Before get double\n");  
  if(!enif_get_double(env, argv[4], &desired_error)) {
    return enif_make_badarg(env);
  }
  fann_train_on_file(ann, file_name, max_epochs, epochs_between_reports,
		     desired_error);
  return enif_make_atom(env, "ok");
}

static ErlNifFunc nif_funcs[] =
{
  {"create_standard", 1, create_standard_nif},
  {"train_on_file", 5, train_on_file_nif},
};

ERL_NIF_INIT(fann,nif_funcs,load,NULL,NULL,NULL)
