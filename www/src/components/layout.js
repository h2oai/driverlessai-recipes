/**
 * Layout component that queries for data
 * with Gatsby's useStaticQuery component
 *
 * See: https://www.gatsbyjs.org/docs/use-static-query/
 */

import React from "react"
import PropTypes from "prop-types"
import Footer from "./footer"
import { Container, Typography } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';

const useStyles = makeStyles(theme => ({
  heroContent: {
    padding: theme.spacing(6, 0, 5),
  },
  yellowTitle: {
    background: 'rgb(255, 229, 43)',
    padding: '5px',
  },
}))

const Layout = ({ children }) => {
  const classes = useStyles()

  return (
    <>
    <Container maxWidth="sm" component="main" className={classes.heroContent}>
        <Typography component="h1" variant="h2" align="center" color="textPrimary" gutterBottom>
          H2O.ai <span className={classes.yellowTitle}>Catalog</span>
        </Typography>
        <Typography variant="h5" align="center" color="textSecondary" component="p">
          Extend the power of Driverless AI with custom recipes and build your own AI!
        </Typography>
    </Container>
    <Container maxWidth="xl" component="main">
            <main>{children}</main>
    </Container>
    <Footer />
    </>
  )
}

Layout.propTypes = {
  children: PropTypes.node.isRequired,
}

export default Layout
