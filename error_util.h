#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

#ifndef _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#endif

#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp
#endif

#else

#include <string.h>
#include <strings.h>
#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

#endif
inline int stringRemoveDelimiter(char delimiter, const char *string)
{
  int string_start = 0;
  
  while(string[string_start] == delimiter)
  {
    string_start++;
  }
  
  if(string_start >= (int)strlen(string)-1)
  {
    return 0;
  }
  
  return string_start;
}

inline int stringRemoveDelimiter(char delimiter, const char *string)
{
  int string_start = 0;
  
  while( string[string_start]==delimiter)
  {
    string_start++;
  }
  
  if(string_start >= (int)strlen(string)-1)
  {
    return 0;
  }
  return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char** argv, const char* string_ref)
{
  bool bFound = false;
  
  if(argc>=1)
  {
    for(int i=1; i<argc; ++i)
    {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      
      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = (int)(equal_pos == 0 ? strlen(string_argv):equal_pos - string_argv);
      
      int length = (int)strlen(string_ref);
      
      if(length == argv_length && !STRNCASECMP(string_argv, string_ref, length))
      {
        bFound = true;
        continue;
      }
    }
  }
  return bFound;
}

