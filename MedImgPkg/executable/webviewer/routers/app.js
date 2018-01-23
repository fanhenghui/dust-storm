const express = require('express');
let router = express.Router();

module.exports = router;

router.get('/', (req,res)=>{
    res.redirect('app/review');
});

router.get('/review', (req, res)=>{
    //check login(check session's user)
    if (!req.session.user) { 
        req.session.error = '请先登录'
        res.redirect('/user/login');
    } else {
        console.log(`user ${req.session.user.name} into review`);
        res.render('review', {
            username: req.session.user.name
        });     
    }
});