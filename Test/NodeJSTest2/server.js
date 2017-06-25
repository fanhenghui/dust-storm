var http = require("http");
var fs = require("fs");
var event = require("events");
var url = require("url");
var util = require("util");
var querystring =require("querystring");

function start(route){
    function onRequest(req , res){

        var body = "";
        req.on("data" ,function(chunk){
            body += chunk;
            console.log("chunk data" + body + "\n");
        });

        req.on("end" , function(chunk){
            body = querystring.parse(body);
            res.writeHead(200 , {"Content-type": "text/html ; charset=utf-8"});

            if(body.name && body.url){
                res.write("网站名：" + body.name);
                res.write("<br>");
                res.write("网站 URL：" + body.url);
            }else {
                var pathname = url.parse(req.url).pathname;
                console.log("request for " + pathname + " received.");
            
                route(pathname);
    
            
                html_content = fs.readFileSync("./index.html").toString();
                res.write("hello node js\n");
                var params = url.parse(req.url , true).query;
                res.write(html_content);
                // res.write("网站名 : " + params.name + "\n");
                // res.write("网站 URL: " + params.url + "\n");
            }
            res.end();
        });

        
    }

    http.createServer(onRequest).listen(8080);

    

    // var t = setTimeout(function(){
    //     console.log( __filename );
    //     console.log(__dirname);    
    // } , 3000);

    // clearTimeout(t);

    // var i = setInterval(function(){
    //     console.log("tick");
    // },1000);

    console.log("server running at http://127.0.0.1:8080/");
    console.log(process.pid);
    console.log(process.arch);
    console.log(process.platform);
    //console.log(process.argv);
    //console.log(process.execPath);
    //console.log(process.env)
    //console.log(process.mainModule);
    console.log(process.getuid());
    console.log(process.getgid());
    console.log(process.memoryUsage());
    console.log(process.memoryUsage().rss);
    console.log(process.memoryUsage().heapTotal);
    console.log(process.memoryUsage().heapUsed);
    console.log(process.memoryUsage().external);

    console.log("process.env.port " + process.env.PORT );
}


exports.start = start;


//var data = fs.readFileSync("test.txt");
// console.log(data.toString());

// fs.readFile("test.txt" , function(err , data){
//     console.log(data.toString());    
// });

// var emitter = new event.EventEmitter();
// emitter.on("someEvent" , function(arg1 , arg2){
//     console.log("listenner1" , arg1 , arg2);
// });

// emitter.on("someEvent" , function(arg1 , arg2){
//     console.log("listenner2" , arg1 , arg2);
// });

// emitter.emit("someEvent" , 1 , 2);
// console.log(event.EventEmitter.listenerCount(emitter , "someEvent"));

// var buf = new Buffer(16);
// for(var i = 0 ; i<8 ; ++i)
// {
//     buf.writeInt16BE(i , i*2);
// }
// for(var i = 0 ; i<8 ; ++i)
// {
//     console.log(buf.readInt16BE(i*2));
// }

// var Hello2 = require("./hello");
// h = new Hello2();
// h.setName("jim");
// h.sayName(); 
