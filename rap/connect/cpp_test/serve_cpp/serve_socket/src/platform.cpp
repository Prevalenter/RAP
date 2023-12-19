#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sstream>
#include "include/timer.h"
#include "include/platform.h"

using namespace std;
//using namespace str;


void sent_cur_states(int cfd){
    char msg_angles[MAXSIZE];
    std::stringstream ss_angles;
    ss_angles << "cur angles: ";
    for(int i=0; i<12; i++){
        ss_angles << i*1.2314 << " ";
    }

    strcpy(msg_angles, ss_angles.str().c_str());
//    cout << ss_angles.str().data() << endl;
    write(cfd, msg_angles, ss_angles.str().length()+1);
}

Platform::Platform(){
//    cout << "construct function" << endl;
    init_socket();
}

void Platform::state_timer_start(){
    std::cout << "--- start period timer ----" << std::endl;
    timer.start(500, std::bind(sent_cur_states, cfd));
//    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
//    timer.stop();
    std::cout << "--- stop period timer ----" << std::endl;
}

void Platform::init_socket(){
    sfd = socket(AF_INET, SOCK_STREAM, 0);

    if(sfd < 0){
        cout << "fail to create TCP/IPV4 socket..." << endl;
        exit(0);
    }

    struct sockaddr_in serve_addr, client_addr;

    memset(&serve_addr, 0, sizeof (serve_addr));
    serve_addr.sin_family = AF_INET;
    serve_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serve_addr.sin_port = htons(IP_PORT);

    if(bind(sfd, (struct sockaddr*)&serve_addr, sizeof (serve_addr)) < 0){
        cout << "bind error ";
        exit(0);
    }

    if(listen(sfd, 20) <0){
        cout << "listen error ";
        exit(0);
    }

    cout << "========== wait for client request ==========" << endl;
    cfd = accept(sfd, (struct sockaddr*)NULL, NULL); // block for the client connet
    if ( cfd < 0 ){
        cout << "accept error" << endl;
    }
    else{
        cout << "client: " << cfd << ", welcome!" << endl;
    }
}

void Platform::run(){
    while(1){
        memset(msg, 0, sizeof (msg));
        nrecv_size = read(cfd, msg, MAXSIZE);
        if( nrecv_size <0 ){
            cout << "accept error " << endl;
            continue;
        }
        else if (nrecv_size==0){
            cout << "client has disconnected!" << endl;
            close(cfd);
            break;
        }
        else{
            cout << "recv msg: " << msg << endl;
//            for(int i=0; msg[i]!='\0'; i++){
//                msg[i] = toupper(msg[i]);
//            }
//            if(write(cfd, msg, strlen(msg)+1)<0)
//            {
//                cout << "accept error" << endl;
//            }
        }

    }
    close(cfd);
    close(sfd);

    cout << "hello" << endl;
}



#define MAXSIZE 1024
#define IP_ADDR "127.0.0.1"
#define IP_PORT 12345




