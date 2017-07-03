// var net = require('net');
// var client = net.connect({port: 8080}, function() {
//    console.log('连接到服务器！');  
// });
// client.on('data', function(data) {
//    console.log(data.toString());
//    client.end();
// });
// client.on('end', function() { 
//    console.log('断开与服务器的连接');
// });


var ipc=require('node-ipc');

ipc.config.id   = 'hello';
ipc.config.retry= 1500;

ipc.connectTo(
    'world0',
    function(){
        ipc.of.world.on(
            'connect',
            function(){
                ipc.log('## connected to world ##'.rainbow, ipc.config.delay);
                ipc.of.world.emit(
                    'message',  //any event or message type your server listens for 
                    'hello'
                )
            }
        );
        ipc.of.world.on(
            'disconnect',
            function(){
                ipc.log('disconnected from world'.notice);
            }
        );
        ipc.of.world.on(
            'message',  //any event or message type your server listens for 
            function(data){
                ipc.log('got a message from world : '.debug, data);
            }
        );
    }
);