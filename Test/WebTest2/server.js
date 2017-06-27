express = require("express");
util = require("util");
var bodyparser = require("body-parser");
var fs = require("fs");
var child_process = require("child_process");

var app = express();

var urlencodedParser =bodyparser.urlencoded({extended:false});

app.use(express.static(__dirname + "/public"));

app.get("/index.html" , function(req , res){
    console.log("hostname : " + req.hostname);
    console.log("ip : " + req.ip);
    console.log("orignal URL : " + req.orignalUrl);
    console.log("base URL : " + req.baseUrl);
    console.log("path : " + req.path);
    console.log("protocol : " + req.protocol);
    console.log("query : " + util.inspect(req.query));
    console.log("route : " + util.inspect(req.route));
    console.log("subdomains : " + req.subdomains);
    console.log("body : " + req.body);

    //res.send("hello world");
    res.sendFile(__dirname + "/" +"index.html");

});

// app.get("/process_get" , function(req , res){

//     var response = { 
//         "username" : req.query.username ,
//         "password" : req.query.password 
//     };
//     console.log(util.inspect(response));
//     res.sendFile(__dirname + "/" + "jump.html");
//     //res.end(JSON.stringify(response));
// });

app.post("/login_in" , urlencodedParser , function(req , res){

    if(req.body.username == "wr" && req.body.password == "000000"){
    
        console.log("login in success.");
        var worker = child_process.spawn('./', ['1']);

        res.sendFile(__dirname + "/" + "jump.html");
    }
    else{
        console.log("login in failed!");
    }
   
});

app.post("/file_upload" , urlencodedParser , function(req , res){
    console.log(req.body.file);

});


app.get("/ok" , function(req , res){

    res.send("OK");
});


app.get("/error" , function(req , res){

    res.send("error");
});

var server  = app.listen(8080 , function(){

    var host = server.address().address;
    var port = server.address().port;

    console.log("address is http://%s:%s" , host , port);
});