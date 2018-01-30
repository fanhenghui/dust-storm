const express = require('express');
let router = express.Router();

const common = require('../libs/common');

module.exports = router;

router.get('/', (req,res)=>{
    res.render('index', {title: `Medical-Imaging [${common.getIPAddress() + ':' + req.connection.localPort}]`});``
});