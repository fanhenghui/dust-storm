express = require("express");
util = require("util");
var bodyparser = require("body-parser");
var fs = require("fs");
var child_process = require("child_process");

console.log("test.js start");

const out = fs.openSync('./out.log', 'a');
const err = fs.openSync('./out.log', 'a');

var worker = child_process.spawn('./worker' ,['1'] ,{detached: true,
  stdio: 'ignore'});

var worker2 = child_process.spawn('./worker' ,['2'] ,{detached:true,
  stdio: 'ignore'});

// worker.stdout.on('data', function (data) {
//       console.log('stdout: ' + data);
//    });

// var workerProcess = child_process.spawn('node', ['support.js', 1]);

// workerProcess.stdout.on('data', function (data) {
//       console.log('stdout: ' + data);
//    });

//    workerProcess.stderr.on('data', function (data) {
//       console.log('stderr: ' + data);
//    });

//    workerProcess.on('close', function (code) {
//       console.log('子进程已退出，退出码 '+code);
//    });
   
console.log("test.js end");