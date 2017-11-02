var express = require('express');
var router = express.Router();

router.get('/', function(req, res, next) {
    // useless code cookies 已经以session的形式发送了
    // if (req.cookies.isVisit) {
    //   console.log('welcome back.');
    //   console.log(req.cookies);
    // } else {
    //   let cookieName = require('./server-ip.js').serverip + new Date().getTime();
    //   res.cookie('isVisit', 1, {maxAge: 60 * 1000});
    //   res.cookie('name', cookieName);
    //   console.log("welcome.");
    // }
    res.render('index', {
        title: 'Web-based Medical Viewer from Baidu'
    }); // 到达此路径则渲染index文件，并传出title值供 index.html使用
});
router.route('/login').get(function(req, res) { // 到达此路径则渲染login文件，并传出title值供 login.html使用
    res.render('login', {
        title: 'User Login'
    });
}).post(function(req, res) {
    var uName = req.body.uname;
    var uPwd = req.body.upwd;
    global.dbHandel.signIn(uName, uPwd, function(res2, err2) {
        if (err2) { 
            //数据库操作错误就返回给原post处（login.html) 状态码为500的错误
            res.status(500).send('数据库操作失败！');
            console.log(err2);
        } else if (res2 == -1) {
            req.session.error = '用户名不存在';
            res.status(404).send('用户名不存在'); //	状态码返回404
        } else if (res2 == -2) {
            req.session.error = '密码错误';
            res.status(404).send('密码错误');
        } else if (res2 == 0) {
            req.session.user = {name:uName};
            res.send(200);
        }
    });
});

router.route('/register').get(function(req, res) {
    res.render('register', {
        title: 'User register'
    });
}).post(function(req, res) {
    var uName = req.body.uname;
    var uPwd = req.body.upwd;
    global.dbHandel.register(uName, uPwd, function(res2, err2) {
        if (err2) {
            //数据库操作错误就返回给原post处（login.html) 状态码为500的错误
            res.status(500).send('数据库操作失败！');
            console.log(err2);
        } else if (res2 == -1) {
            req.session.error = '用户名已存在！';
            res.status(500).send('用户名已存在！');
        } else if (res2 == 0) {
            req.session.user = {name:uName};
            res.send(200);
        }
    });
});

router.get('/home', function(req, res) {
    if (!req.session.user) { //到达/home路径首先判断是否已经登录
        req.session.error = '请先登录'
        res.redirect('/login'); //未登录则重定向到 /login 路径
    }
    res.render(
        'home', {
            title: 'Home',
            name: req.session.user.name
        }); //已登录则渲染home页面
});

router.get('/logout', function(req, res) { // 到达 /logout 路径则登出
    // session中user,error对象置空，并重定向到根路径
    req.session.user = null;
    req.session.error = null;
    res.redirect('/');
});

router.route('/review').get(function(req, res) {
    if (!req.session.user) { //到达/home路径首先判断是否已经登录
        req.session.error = '请先登录'
        res.redirect('/login'); //未登录则重定向到 /login 路径
    } else {
        console.log('login name is : ', req.session.user.name);
        var ip = require('./server-ip.js').serverip;
        //send user name and server ip
        res.render('review', {
            username: req.session.user.name,
            serverip: ip + ':' + global.appPort
        }); // go directly to review page
    }
});
module.exports = router;
