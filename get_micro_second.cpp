////
//// get_micro_second.cpp: the function for measuring execution time
////

///
/// The standard include files
///
#include <iostream>
#include <sys/time.h>
#include <unistd.h>

#include <cstdlib>

///
/// Get time using the function gettimeofday()
///
double getMicroSecond( void )
{

    double sec;

    struct timeval timev;      // time value
    struct timezone timez;     // time zone

    if( gettimeofday( &timev, &timez ) == -1 ) {
	std::cerr << "Could not get time by gettimeofday()." << std::endl;
	exit(1);
    }

    // the unit of returned value is second
    sec = static_cast<double>(timev.tv_sec) + static_cast<double>(timev.tv_usec) * 1e-6;

    return sec;

}
