
#ifndef MULTIRESOLUTIONIMAGEINTERFACE_EXPORT_H
#define MULTIRESOLUTIONIMAGEINTERFACE_EXPORT_H

#ifdef WIN32
    #ifdef MULTIRESOLUTIONIMAGEINTERFACE_STATIC_DEFINE
    #  define MULTIRESOLUTIONIMAGEINTERFACE_EXPORT
    #  define MULTIRESOLUTIONIMAGEINTERFACE_NO_EXPORT
    #else
    #  ifndef MULTIRESOLUTIONIMAGEINTERFACE_EXPORT
    #    ifdef multiresolutionimageinterface_EXPORTS
            /* We are building this library */
    #      define MULTIRESOLUTIONIMAGEINTERFACE_EXPORT __declspec(dllexport)
    #    else
            /* We are using this library */
    #      define MULTIRESOLUTIONIMAGEINTERFACE_EXPORT __declspec(dllimport)
    #    endif
    #  endif
    
    #  ifndef MULTIRESOLUTIONIMAGEINTERFACE_NO_EXPORT
    #    define MULTIRESOLUTIONIMAGEINTERFACE_NO_EXPORT 
    #  endif
    #endif
    
    #ifndef MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED
    #  define MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED __declspec(deprecated)
    #endif
    
    #ifndef MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED_EXPORT
    #  define MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED_EXPORT MULTIRESOLUTIONIMAGEINTERFACE_EXPORT MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED
    #endif
    
    #ifndef MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED_NO_EXPORT
    #  define MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED_NO_EXPORT MULTIRESOLUTIONIMAGEINTERFACE_NO_EXPORT MULTIRESOLUTIONIMAGEINTERFACE_DEPRECATED
    #endif
    
    #if 0 /* DEFINE_NO_DEPRECATED */
    #  ifndef MULTIRESOLUTIONIMAGEINTERFACE_NO_DEPRECATED
    #    define MULTIRESOLUTIONIMAGEINTERFACE_NO_DEPRECATED
    #  endif
    #endif
#else
    #  define MULTIRESOLUTIONIMAGEINTERFACE_EXPORT
    #  define MULTIRESOLUTIONIMAGEINTERFACE_NO_EXPORT
#endif

#endif
