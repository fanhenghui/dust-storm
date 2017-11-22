const express = require('express');
const router = express.Router();

router.get(global.subPage + '/', (req, res, next)=> {
    res.redirect('login');
});

router.route(global.subPage+'/login')
.get((req, res)=>{
    res.render('login', {
        subpage: global.subPage,
        title: 'User Login'
    });
})
.post((req, res)=>{
    let uName = req.body.uname;
    let uPwd = req.body.upwd;
    global.dbHandel.signIn(uName, uPwd, (res2, err2)=>{
        if (err2) { 
            //server error
            res.status(500).send('数据库操作失败！');
            console.log(err2);
        } else if (res2 == -1) {
            req.session.error = '用户名不存在';
            res.status(404).send('用户名不存在');
        } else if (res2 == -2) {
            req.session.error = '密码错误';
            res.status(404).send('密码错误');
        } else if (res2 == 0) {
            req.session.user = {name:uName};
            res.send(200);
        }
    });
});

router.route(global.subPage+'/register')
.get((req, res)=>{
    res.render('register', {
        subpage: global.subPage,
        title: 'User register'
    });
})
.post((req, res)=>{
    let uName = req.body.uname;
    let uPwd = req.body.upwd;
    global.dbHandel.register(uName, uPwd, (res2, err2)=>{
        if (err2) {
            //server error
            res.status(500).send('数据库操作失败！');
            console.log(err2);
        } else if (res2 == -1) {
            req.session.error = '用户名已存在！';
            res.status(500).send('用户名已存在！');
        } else if (res2 == 0) {
            req.session.user = {name:uName};//record to session
            res.send(200);
        }
    });
});

router.get(`${global.subPage}/logout`, (req, res)=>{
    // clear session's user/error and redirect to login
    req.session.user = null;
    req.session.error = null;
    res.redirect(global.subPage+'/login');
});

router.route(global.subPage+'/review').get((req, res)=>{
    //check login(check session's user)
    if (!req.session.user) { 
        req.session.error = '请先登录'
        res.redirect(global.subPage+'/login');
    } else {
        console.log(`user ${req.session.user.name} into review`);
        res.render('review', {
            subpage: global.subPage,
            username: req.session.user.name
        });     
    }
});

module.exports = router;
