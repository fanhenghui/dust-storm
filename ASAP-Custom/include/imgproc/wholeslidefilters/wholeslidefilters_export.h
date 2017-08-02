
#ifndef WHOLESLIDEFILTERS_EXPORT_H
#define WHOLESLIDEFILTERS_EXPORT_H

#ifdef WHOLESLIDEFILTERS_STATIC_DEFINE
#  define WHOLESLIDEFILTERS_EXPORT
#  define WHOLESLIDEFILTERS_NO_EXPORT
#else
#  ifndef WHOLESLIDEFILTERS_EXPORT
#    ifdef wholeslidefilters_EXPORTS
        /* We are building this library */
#      define WHOLESLIDEFILTERS_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define WHOLESLIDEFILTERS_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef WHOLESLIDEFILTERS_NO_EXPORT
#    define WHOLESLIDEFILTERS_NO_EXPORT 
#  endif
#endif

#ifndef WHOLESLIDEFILTERS_DEPRECATED
#  define WHOLESLIDEFILTERS_DEPRECATED __declspec(deprecated)
#endif

#ifndef WHOLESLIDEFILTERS_DEPRECATED_EXPORT
#  define WHOLESLIDEFILTERS_DEPRECATED_EXPORT WHOLESLIDEFILTERS_EXPORT WHOLESLIDEFILTERS_DEPRECATED
#endif

#ifndef WHOLESLIDEFILTERS_DEPRECATED_NO_EXPORT
#  define WHOLESLIDEFILTERS_DEPRECATED_NO_EXPORT WHOLESLIDEFILTERS_NO_EXPORT WHOLESLIDEFILTERS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef WHOLESLIDEFILTERS_NO_DEPRECATED
#    define WHOLESLIDEFILTERS_NO_DEPRECATED
#  endif
#endif

#endif