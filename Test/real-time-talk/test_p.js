// var net = require('net');
// var server = net.createServer(function(connection) { 
//    console.log('client connected');
//    connection.on('end', function() {
//       console.log('客户端关闭连接');
//    });
//    connection.write('Hello World!\r\n');
//    connection.pipe(connection);
// });
// server.listen("hellowr", function() { 
//   console.log('server is listening');
// });

for(var i =0 ; i<2 ; ++i)
{
    var ipc=require('node-ipc');
 
    ipc.config.id   = 'world' + i;
    ipc.config.retry= 1500;

    ipc.serve(
        function(){
            ipc.server.on(
                'message',
                function(data,socket){
                    ipc.log('got a message : '.debug, data);
                    ipc.server.emit(
                    socket,
                    'message',  //this can be anything you want so long as 
                                //your client knows. 
                    data+' world!'
                );
            }
        );
        ipc.server.on(
            'socket.disconnected',
            function(socket, destroyedSocketID) {
                ipc.log('client ' + destroyedSocketID + ' has disconnected!');
            }
        );
    }
);

ipc.server.start();
}
