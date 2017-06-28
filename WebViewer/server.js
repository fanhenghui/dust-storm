//用sockio来构造server（本质上还是HTTPserver）
var app = require("express")();
var http = require("http").Server(app);
var io = require("socket.io")(http);
var util = require("util");
var fs = require("fs");
var child_process = require("child_process");
var body_parser = require("body-parser");

//添加静态目录
app.use(require("express").static(__dirname + "/public"));

var uRLEncoderParser = body_parser.urlencoded({extended:false});

//主页的响应
app.get("/index.html" , function(req, res){
    res.sendFile(__dirname + "/index.html");
    console.log("get index.html");
});

//登录表单的提交
// app.post("/login_in" , uRLEncoderParser , function(req , res){
//     //这里查数据库，匹配用户名和密码
//     if(req.body.username == "wr" && req.body.password == "000000")
//     {
//         res.sendFile(__dirname + "/review.html");
//     }
//     console.log(req.body.username + " has login in.");

//     //建立websocket长连接
// });


//Websocket 长连接用户
var onlineUsers = {};
var onLineCount = 0;

io.on("connection" , function(socket){
    console.log("a user is connected");

    //监听用户加入
    socket.on("login" , function(obj){
        if(obj.username == "")
        {
            console.log("invalid null username!");
            return;
        }

        console.log(obj.username + " has login.");
        console.log("id : " + obj.id);


    });

    //监听用户退出
    socket.on("disconnect" , function(obj){
        console.log(obj.username + " has disconnect.");

    });


    //监听用户消息
    socket.on("message" , function(obj){
        console.log(obj.username + " message : " + obj.content);
    });
})



var server = http.listen(8080,function(){
    var address = server.address();
    console.log("address is " , util.inspect(address));
});
