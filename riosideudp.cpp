#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>

#define BUFLEN 512
#define VISION_PORT 5800

struct sockaddr_in local, remote;
int retval, sockfd, s, i, len;
socklen_t remote_len;
char buffer[BUFLEN];
int nonBlocking = 1;
int opt_val = 1;

struct target
{
	int version; // packet version
	int valid;		 // status of data
	int angle;		 // -32767 to 32767
	int distance;	 // -32767 to 32767
	int empty1;		 // -32767 to 32767
	int empty2;		 
	int empty3;		
  int empty4;
};



target vision_target;
int vision_timer;

///////////////////////////////////////////////////////////////////////////////////////////
//sets up the socket to listen on
void Vision_UDP_Init()
{
	sockfd = socket(AF_INET, SOCK_DGRAM, 0);
	memset((char *)&local, 0, sizeof(local));
	local.sin_family = AF_INET;
	local.sin_addr.s_addr = htonl(INADDR_ANY);
	local.sin_port = htons(VISION_PORT);
	if (bind(sockfd, (struct sockaddr *)&local, sizeof(local)) < 0)
	{
		perror("bind failed");
		// return;
	}
	fcntl(sockfd, F_SETFL, O_NONBLOCK, nonBlocking);
	opt_val = 1;
	setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
}
