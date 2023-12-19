#include <iostream>
#include <cstring>

#include "include/timer.h"
#include "include/platform.h"

using namespace std;

int main()
{
    Platform pltf;
//    pltf.init_socket();
    pltf.state_timer_start();
    pltf.run();
}

 namespace str;

#define MAXSIZE 1024
#define IP_ADDR "127.0.0.1"
#define IP_PORT 12345


void print_hello(int msg){
    cout << "hell from function: " << msg << endl;
}

void sent_cur_angles(int cfd){
//    double angles[6];
    char msg_angles[MAXSIZE];
    std::stringstream ss_angles;
    ss_angles << "cur angles: ";
    for(int i=0; i<6; i++){
//        angles[i] = double(i)*1.2314;
        ss_angles << i*1.2314 << " ";
    }

//    cout << ss_angles.str().c_str() << endl;
//    const char * msg_angles = "sent test";

//    const char * msg_angles = ss_angles.str().data();
    strcpy(msg_angles, ss_angles.str().c_str());
//    msg_angles[ss_angles.str()] = ;
    cout << ss_angles.str().data() << endl;
    write(cfd, msg_angles, ss_angles.str().length()+1);
}

int main()
{


    int sfd, cfd; //sfd short for server file descriptor
    char msg[MAXSIZE];
    int nrecv_size = 0;

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

    Timer timer;

    std::cout << "--- start period timer ----" << std::endl;
    timer.start(500, std::bind(sent_cur_angles, cfd));
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    timer.stop();
    std::cout << "--- stop period timer ----" << std::endl;

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
            for(int i=0; msg[i]!='\0'; i++){
                msg[i] = toupper(msg[i]);
            }
            if(write(cfd, msg, strlen(msg)+1)<0)
            {
                cout << "accept error" << endl;
            }
        }

    }
    close(cfd);
    close(sfd);

    cout << "hello" << endl;
}
