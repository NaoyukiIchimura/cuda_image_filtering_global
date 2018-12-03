////
//// path_hander.cpp: Handling paths of files
////

///
/// The standard include files
///
#include <iostream>
#include <string>

///
/// The functions for handling the path of a file
///
void getDirFileName( const std::string &path, std::string *dir, std::string *file )
{

    std::string::size_type p = path.find_last_of( "/" ); 

    if( p == std::string::npos ) { // no / is the path
	*dir = ".";
	*file = path;
    } else if( p == 0 ) { // the first character of the path is /
	*dir = ".";
	*file = path.substr( p + 1 );
    } else { // found / in the path
	*dir = path.substr( 0, p ); 
	*file = path.substr( p + 1 );
    }

}

void getPrefixExtension( const std::string &fileName, std::string *prefix, std::string *extension )
{

    std::string::size_type p = fileName.find_last_of( "." );

    if( p == std::string::npos ) { // no . in the file name
	*prefix = fileName;
	*extension = ""; // empty
    } else {	// found . in the file name
	*prefix = fileName.substr( 0, p );
	*extension = fileName.substr( p + 1 );
    }

}
