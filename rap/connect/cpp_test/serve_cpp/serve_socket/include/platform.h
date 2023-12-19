#ifndef PLATFORM_H
#define PLATFORM_H

#define MAXSIZE 1024
#define IP_ADDR "127.0.0.1"
#define IP_PORT 12345


class Platform{
public:
    void init_socket();
    double * get_rbt_states(); // get the state of joint and endeffector
    void state_timer_start();
    void state_timer_stop();
//    void sent_cur_states();
    void run();

    Platform(); // initial function
//    init_socket();

    int sfd, cfd; //sfd short for server file descriptor
    char msg[MAXSIZE];
    int nrecv_size = 0;
    Timer timer;


private:
    int a;
};


#endif // PLATFORM_H
