


int TestIPCProxy(int argc , char* argv[])
{
    _message_queue.activate();

    boost::thread th(run);

    boost::thread th2(mainloop);

    th.join();
    th2.join();

    return 0;
}